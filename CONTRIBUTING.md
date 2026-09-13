# Contributing to TerraGPU

TerraGPU focuses on GPU acceleration of geospatial processing. Discuss substantial
changes in a GitHub issue or draft pull request so maintainers can review the
scientific assumptions, scope and validation plan. Routine bug fixes can start
with a reproducer and a pull request. Follow the code of conduct below.

## Set up a development environment

From a checkout, with uv installed:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test,parallel,pace,viirs,benchmark,publication]'
.venv/bin/python -m pytest -q
.venv/bin/python -m build
```

This installs CPU development and plotting dependencies. The base package keeps
Dask, CuPy and product-specific optional dependencies separate. Do not make a
CPU import require CUDA. Do not add machine-learning frameworks or restore the
former raster-wrapper dependency. Supported Python versions are declared in
[pyproject.toml](pyproject.toml); CI currently tests Python 3.11, 3.12 and 3.13.

The current tests live in `tests/` and use pytest. GPU tests are marked `gpu` and
require a functioning CUDA/CuPy environment. A skipped GPU test is not a CUDA pass.
For PRISM, use the architecture-aware setup scripts and environment recorded in
[the PRISM checklist](docs/prism-validation.md) and
[publication run instructions](docs/publication-runs.md).

## Scientific and performance changes

- State the product, units, band mapping, scale/offset, nodata/QA policy and
  coordinate convention. Preserve affine rasters or native swaths as appropriate.
- Provide an independent numerical reference and tests for masks, georeferencing,
  boundaries and relevant malformed input. CPU/GPU agreement alone does not
  establish that a scientific algorithm is correct.
- Keep Dask optional. Compare scheduling strategies on equivalent work and output
  layouts; a new dependency is not itself evidence of better performance.
- Report the timing scope. Separate resident computation, prepared-cache I/O and
  native-input-to-output processing. Include required preparation/output costs in
  end-to-end claims, synchronize CUDA, retain raw trials and disclose cache policy.
- Record source revision, dependency versions, input checksums, worker counts,
  hardware and numerical error. Label sampled memory accurately. A cuFile call
  is not proof of direct-storage transfers.

See [the roadmap](docs/modernization-plan.md),
[execution backends](docs/execution-backends.md), and
[GPU I/O experiment](docs/gpu-direct-io.md) for current assumptions and limits.

## Pull request review

Explain the problem, resulting behavior, scientific conventions and validation.
Use a focused regression test for a bug; run the tests affected by the change and
required CI checks. Include GPU evidence only if it was actually collected.
Document changed APIs, CLI arguments and environment variables in the relevant
Markdown guide. Current source documentation and Markdown guides take precedence
over historical generated HTML; regenerating that HTML with obsolete commands
is not a release requirement.

Keep unrelated edits out of the pull request. Summarize user-visible changes in
[CHANGELOG.md](CHANGELOG.md) when appropriate. Maintainers coordinate version
changes for releases; contributors should not independently bump every example
for each pull request. Follow the repository's current version in pyproject.toml.
No merge, release, archive publication or performance qualification follows
merely from opening a pull request or passing CPU CI.

## Data and reproducibility

Use small synthetic fixtures or documented public downloads. Never commit
credentials, tokens, proprietary imagery, private machine paths or raw system
logs to public GitHub. Review result artifacts before publishing them: environment
and mount logs can contain private details even when the imagery is public.
Keep local data, caches, environments and result archives outside tracked source.

Freeze the benchmark revision and environment for a run series. Do not pull
unrelated changes midway through an independent-job comparison. An analysis-only
update can be applied later to archived reports; record both benchmark and
analysis revisions. Share numerical failures and negative performance results.
The [manuscript scaffold](paper/manuscript.md) lists the remaining publication
gates and the AI-assistance disclosure that authors must review.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at [INSERT EMAIL ADDRESS]. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
