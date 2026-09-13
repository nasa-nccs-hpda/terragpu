"""Explicit, cached example downloads. Importing TerraGPU never downloads data."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile
import ssl
import certifi
from urllib.request import urlopen

SAMPLE = {
    'name': 'rasterio-rgb',
    'filename': 'RGB.byte.tif',
    'url': 'https://raw.githubusercontent.com/rasterio/rasterio/57d9fda6c31c5595ea54262f905b43c5f8419e06/tests/data/RGB.byte.tif',
    'sha256': 'd7cbe932c7ed74a627706a9e9df99f706df3e5abc7d45a49e9d00677a6b09eb4',
    'size_bytes': 1745956,
    'bands': ['red', 'green', 'blue'],
    'units': 'display-scaled uint8; not surface reflectance; no NIR band',
    'source': 'Rasterio test dataset, RGB.byte.tif',
    'license_url': 'https://github.com/rasterio/rasterio/blob/57d9fda6c31c5595ea54262f905b43c5f8419e06/LICENSE.txt',
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write_json(path, value):
    """Publish complete metadata atomically, not a partial JSON file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.manifest-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch_sample(cache_dir='data/examples'):
    """Fetch a pinned 1.7 MB public RGB raster; reuse verified bytes offline."""
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SAMPLE['filename']
    if not (path.is_file() and path.stat().st_size == SAMPLE['size_bytes'] and sha256(path) == SAMPLE['sha256']):
        fd, temporary = tempfile.mkstemp(dir=directory, prefix='.download-')
        try:
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=certifi.where())
            with os.fdopen(fd, 'wb') as stream, urlopen(SAMPLE['url'], timeout=60, context=context) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > SAMPLE['size_bytes']:
                        raise ValueError('Sample exceeds its pinned size')
                    stream.write(block)
            if total != SAMPLE['size_bytes'] or sha256(temporary) != SAMPLE['sha256']:
                raise ValueError('Sample checksum/size mismatch; cached file was not replaced')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _write_json(directory / 'sample-manifest.json', {'schema_version': 1, **SAMPLE})
    return path


def _earthaccess():
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError('Install terragpu[data] for NASA discovery/downloads') from exc
    return earthaccess


def fetch_worldview_sample(cache_dir='data/worldview-example'):
    """Download a pinned vendor sample (~257 MB); preserve proprietary attribution.

    This public sample does not grant redistribution rights to licensed imagery.
    Verified files are reusable offline; interrupted downloads never replace them.
    """
    from .sample_data import WORLDVIEW_SAMPLE
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    for item in WORLDVIEW_SAMPLE['files']:
        path = directory / item['name']
        if path.is_file() and path.stat().st_size == item['size_bytes'] and sha256(path) == item['sha256']:
            continue
        fd, temporary = tempfile.mkstemp(dir=directory, prefix='.download-')
        try:
            with os.fdopen(fd, 'wb') as stream, urlopen(item['url'], timeout=60, context=context) as response:
                total = 0
                while block := response.read(1024*1024):
                    total += len(block)
                    if total > item['size_bytes']:
                        raise ValueError('WorldView sample exceeds pinned size')
                    stream.write(block)
            if total != item['size_bytes'] or sha256(temporary) != item['sha256']:
                raise ValueError('WorldView sample checksum/size mismatch')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _write_json(directory / 'sample-manifest.json', {'schema_version': 1, **WORLDVIEW_SAMPLE})
    return directory / '1040010025C68500.json'


def fetch_satstereo(cache_dir='data/satstereo'):
    """Pinned Purdue SatStereo MP1 sample, with reference disparity/building masks."""
    import tarfile
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'sample.tar.gz'
    url = 'https://engineering.purdue.edu/RVL/Database/SatStereo/database/RVL_SatStereo_Sample_MP1.tar.gz'
    digest = '700398ac6c8b918c8359afc94422905fd74944404ea397dc1d7806fa74e44217'
    if not archive.exists() or sha256(archive) != digest:
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        fd, temporary = tempfile.mkstemp(dir=directory, prefix='.download-')
        try:
            with os.fdopen(fd, 'wb') as stream, urlopen(url, timeout=60, context=context) as response:
                size = 0
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > 17906618:
                        raise ValueError('Stereo sample exceeds pinned size')
                    stream.write(block)
            if sha256(temporary) != digest:
                raise ValueError('Stereo sample checksum mismatch')
            os.replace(temporary, archive)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    # Re-extract verified bytes so modified working inputs cannot enter a run.
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 100_000_000 or any(
                not (m.isfile() or m.isdir()) or not (directory/m.name).resolve().is_relative_to(directory.resolve())
                for m in members):
            raise ValueError('Unexpected stereo archive contents')
        bundle.extractall(directory, filter='data')
    root = directory / 'RVL_SatStereo_Sample_MP1'
    _write_json(directory/'sample-manifest.json', dict(source=url, archive_sha256=digest,
                license='GPL-3.0 per provider download page',
                citation='Patil et al. (2019), A New Stereo Benchmarking Dataset for Satellite Images, arXiv:1907.04404',
                files=[dict(name=str(p.relative_to(directory)), sha256=sha256(p)) for p in sorted(root.rglob('*')) if p.is_file()]))
    return root


def _query(short_name, version, bbox, start, end, limit):
    if not short_name or not version:
        raise ValueError('Supply a collection short name and version')
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('bbox must be west south east north; split antimeridian queries')
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError('start must be no later than end')
    if not 1 <= limit <= 10:
        raise ValueError('Example download limit must be between 1 and 10 granules')
    return dict(short_name=short_name, version=version, bounding_box=list(bbox),
                temporal=[start, end], count=limit)


def _granule_summary(granule):
    return {'concept_id': granule.get('meta', {}).get('concept-id'),
            'granule_ur': granule.get('umm', {}).get('GranuleUR')}



def _login(client, strategy, token_file=None):
    """Use a token file without placing its contents in arguments or manifests.

    Environment mutation is limited to login and restored afterwards. Call this
    from a single controlling thread; earthaccess retains its authenticated session.
    """
    if token_file is None:
        return client.login(strategy=strategy, persist=False)
    if strategy != 'environment':
        raise ValueError('token_file requires the environment login strategy')
    path = Path(token_file).expanduser()
    if os.name == 'posix' and path.stat().st_mode & 0o077:
        raise ValueError('Token file must be private to its owner (chmod 600)')
    token = path.read_text().strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError('Token file must contain only one nonempty token')
    previous = os.environ.get('EARTHDATA_TOKEN')
    try:
        os.environ['EARTHDATA_TOKEN'] = token
        try:
            return client.login(strategy='environment', persist=False)
        except Exception:
            # Do not propagate third-party exception messages containing secrets.
            raise RuntimeError('Earthdata token authentication failed; check token validity and authorization') from None
    finally:
        if previous is None:
            os.environ.pop('EARTHDATA_TOKEN', None)
        else:
            os.environ['EARTHDATA_TOKEN'] = previous


def nasa_data(*, short_name, version, bbox, start, end, limit=1,
              output='data/nasa-example', search_only=False, login_strategy='environment', token_file=None,
              granule_name=None):
    """Discover/download whole native NASA granules, retaining a local manifest.

    Search is anonymous. Downloads require Earthdata credentials supplied to
    earthaccess; this function never persists credentials or saves access URLs.
    A bounding box selects granules, not a spatial subset of each file. Explicit
    dates/version/limit bound the query; a single granule can still be large.
    Reuse a verified completed output offline, or use a new directory for a new query.
    """
    query = _query(short_name, version, bbox, start, end, limit)
    if granule_name is not None:
        query['granule_name'] = granule_name
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if not search_only and output.exists():
        if not manifest_path.is_file():
            raise FileExistsError('Use a new output directory or a completed TerraGPU cache')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('query') != query:
            raise ValueError('Cached query differs; choose a new output directory')
        if not manifest.get('files'):
            raise ValueError('Cached manifest contains no files')
        for item in manifest['files']:
            path = output / item['name']
            if path.resolve().parent != output.resolve():
                raise ValueError('Invalid cached filename')
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError('Cached file missing/corrupt; use a new output directory')
        return manifest
    client = _earthaccess()
    granules = client.search_data(**{**query, 'bounding_box': tuple(query['bounding_box']),
                                     'temporal': tuple(query['temporal'])})
    if not granules:
        raise ValueError('No granules found; adjust dates, region or collection/version')
    summary = {'schema_version': 1, 'query': query,
               'granules': [_granule_summary(g) for g in granules]}
    if search_only:
        return summary
    if login_strategy not in {'environment', 'netrc', 'interactive'}:
        raise ValueError('Unsupported login strategy')
    auth = _login(client, login_strategy, token_file)
    if not auth.authenticated:
        raise RuntimeError('Earthdata authentication required; authenticate locally, not in chat')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.nasa-download-') as temporary:
        staging = Path(temporary) / 'dataset'
        staging.mkdir()
        paths = client.download(granules, local_path=staging, threads=2)
        if not paths:
            raise RuntimeError('NASA returned no downloaded files')
        files = []
        for raw in paths:
            path = Path(raw)
            if path.resolve().parent != staging.resolve() or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('Unexpected or empty downloaded file')
            files.append({'name': path.name, 'size_bytes': path.stat().st_size, 'sha256': sha256(path)})
        manifest = {**summary, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                    'earthaccess_version': client.__version__, 'files': files}
        _write_json(staging / 'manifest.json', manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sample = sub.add_parser('sample', help='Download the no-login RGB example')
    sample.add_argument('--cache-dir', default='data/examples')
    worldview = sub.add_parser('worldview', help='Download the pinned public WorldView-3 ARD sample')
    worldview.add_argument('--cache-dir', default='data/worldview-example')
    stereo = sub.add_parser('satstereo', help='Download the pinned satellite stereo benchmark sample')
    stereo.add_argument('--cache-dir', default='data/satstereo')
    nasa = sub.add_parser('nasa', help='Discover or download native NASA granules')
    nasa.add_argument('--short-name', required=True)
    nasa.add_argument('--version', required=True)
    nasa.add_argument('--bbox', type=float, nargs=4, required=True, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    nasa.add_argument('--start', required=True)
    nasa.add_argument('--end', required=True)
    nasa.add_argument('--limit', type=int, default=1)
    nasa.add_argument('--output', default='data/nasa-example')
    nasa.add_argument('--search-only', action='store_true')
    nasa.add_argument('--token-file', type=Path, help='Private file containing only an Earthdata User Token')
    nasa.add_argument('--login-strategy', choices=['environment', 'netrc', 'interactive'], default='environment')
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        if command == 'sample':
            print(fetch_sample(**args))
        elif command == 'worldview':
            print(fetch_worldview_sample(**args))
        elif command == 'satstereo':
            print(fetch_satstereo(**args))
        else:
            print(json.dumps(nasa_data(**args), indent=2))
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
