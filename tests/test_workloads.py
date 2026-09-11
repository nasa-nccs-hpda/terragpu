import numpy as np
import pytest

pytest.importorskip('scipy')
from terragpu.workloads import focal_mean, spectral_angle, stereo_census, disparity_metrics


def test_focal_nan_edges():
    a=np.arange(30,dtype='float32').reshape(5,6);a[1:3,2]=np.nan
    expected=np.array([[np.nanmean(a[max(0,y-1):y+2,max(0,x-1):x+2]) for x in range(6)] for y in range(5)])
    np.testing.assert_allclose(focal_mean(a,3),expected,atol=2e-6)
    assert np.isnan(focal_mean(np.full((5,5),np.nan,dtype='float32'),3)).all()


def test_angle_known_geometry():
    a=np.array([[[1,0],[0,1],[0,0],[np.nan,1]]],dtype='float32')
    np.testing.assert_allclose(spectral_angle(a,np.array([1,0],dtype='float32')),
                               [[0,np.pi/2,np.nan,np.nan]],atol=1e-7,equal_nan=True)


@pytest.mark.parametrize('shift',[-3,2])
def test_stereo_known_shift_and_invalid_support(shift):
    a=np.random.default_rng(4).random((25,35)).astype('float32');b=np.full_like(a,np.nan)
    if shift>0:b[:,shift:]=a[:,:-shift]
    else:b[:,:shift]=a[:,-shift:]
    result=stereo_census(a,b,-4,4)
    np.testing.assert_equal(result[5:-5,8:-8],shift)
    assert np.isnan(result[:4]).all()
    metrics=disparity_metrics(result,np.full_like(a,shift))
    assert 0<metrics['coverage']<1


@pytest.mark.gpu
def test_complex_gpu_parity():
    cp=pytest.importorskip('cupy')
    a=np.random.default_rng(5).random((20,30)).astype('float32');b=a.copy()
    np.testing.assert_allclose(cp.asnumpy(focal_mean(cp.asarray(a),3,xp=cp)),focal_mean(a,3),atol=1e-6)
    np.testing.assert_equal(cp.asnumpy(stereo_census(cp.asarray(a),cp.asarray(b),-3,3,xp=cp)),stereo_census(a,b,-3,3))
    cube=np.random.default_rng(6).random((5,7,8)).astype('float32');ref=np.arange(8,dtype='float32')+1
    np.testing.assert_allclose(cp.asnumpy(spectral_angle(cp.asarray(cube),cp.asarray(ref),xp=cp)),spectral_angle(cube,ref),atol=2e-6)
