"""Optional CPU runtime, isolated from the ordinary forecast process.

Trusted assets contain the MIT-licensed upstream code, fixed checkpoint and
calibration files. No notebook paths or remote pickle downloads are used.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

CHECKPOINT_SHA='512b244e37cdf1370accd46f4ae969d146cd221506dc253edb69ebb23b0e366a'


def digest(path):
    with open(path,'rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


class ProswinRuntime:
    def __init__(self,root):
        self.root=Path(root)
        manifest=json.loads((self.root/'manifest.json').read_text())
        for name,sha in manifest['sha256'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()) or digest(path)!=sha: raise ValueError('Corrupt PROSWIN assets')
        checkpoint=self.root/'171-211_fold_01.pth'
        if digest(checkpoint)!=CHECKPOINT_SHA: raise ValueError('Untrusted PROSWIN checkpoint')
        sys.path[:0]=[str(self.root/'vendor/proswin/src'),str(self.root/'vendor/solar_image_processing/src')]
        import torch
        from proswin.model.solar_swin_transformer import SolarSwinTransformer
        torch.set_num_threads(4)
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=SolarSwinTransformer(use_pretrained_image_encoder_weights=False,
            distribution_transformer=c['distribution_transformer'],physical_feature_scaler=c['physical_feature_scaler'],**c['hyperparameters'])
        self.model.load_state_dict(c['state_dict'],strict=True);self.model.eval()
        self.calibration={}
        for wl in ['171','211']:
            with (self.root/'calibration'/f'psf_{wl}_1024x1024.pickle').open('rb') as f: psf=pickle.load(f)
            with (self.root/'calibration/degradation_correction_table.pickle').open('rb') as f: correction=pickle.load(f)
            self.calibration[wl]=(psf,correction)

    def crop(self,path,wl,slot):
        import sunpy
        import os
        # SunPy has a separate data-manager path, independent of DOWNLOADDIR.
        sunpy.config.set("downloads", "remote_data_manager_dir",
                         os.environ.get("SUNPY_DOWNLOADDIR", "/tmp/proswin-sunpy-data"))
        import sunpy.map
        import astropy.units as u
        from aiapy.calibrate import correct_degradation
        from solar_image_processing.preprocessing.aia_preprocessor import AIAPreprocessor
        from solar_image_processing.preprocessing.preprocessing_functions import scale_solar_disk_radius
        from skimage.measure import block_reduce
        from skimage.transform import resize
        m=sunpy.map.Map(path); observed=pd.Timestamp(m.date.to_datetime()).tz_localize('UTC')
        if (int(m.meta['wavelnth'])!=int(wl) or int(m.meta['quality']) & ~(1<<30) or
            float(m.meta['lvl_num'])!=1.5 or m.data.shape!=(1024,1024) or
            not np.isfinite(m.data).all() or m.exposure_time.to_value(u.s)<=0 or
            abs(observed-slot)>pd.Timedelta(minutes=1)): raise ValueError('Unsupported PROSWIN image')
        psf,correction=self.calibration[wl]
        engine=AIAPreprocessor(None,psf,correction,{'use_gpu':False})
        a=scale_solar_disk_radius(engine._deconvolve(m),rsun_target=976.)
        a=correct_degradation(a,correction_table=correction)
        image=np.flipud(a.data/a.exposure_time.to_value(u.s))
        reduced=block_reduce(image,(2,2),np.sum)
        return resize(reduced[106:406,106:406],(224,224),order=3,mode='constant',cval=0).astype('float32')

    def predict(self,images,features):
        import torch
        from scipy.stats import lognorm
        from proswin.data_management.solar_dataset import SolarDataset
        x=[SolarDataset._normalize_aia(None,images[wl],cap) for wl,cap in [('171',6457.5),('211',6539.)]]
        f=np.asarray(features,dtype=np.float32)
        if f.shape!=(63,) or not np.isfinite(f).all(): raise ValueError('Expected 63 finite features')
        f=self.model.physical_feature_scaler.transform(f[None,:])
        with torch.inference_mode():
            mu,sigma=self.model({'images':torch.tensor(np.stack([x[0],x[1],x[1]])[None,:],dtype=torch.float32),
                'physical_features':torch.tensor(f,dtype=torch.float32)})
        par=self.model.distribution_transformer.inverse_transform_prediction(pd.DataFrame({'mu':mu.numpy().ravel(),'sigma':sigma.numpy().ravel()})).iloc[0]
        result=float(lognorm.mean(par['shape'],loc=par['loc'],scale=par['scale']))
        if not np.isfinite(result) or result<=0: raise ValueError('Invalid PROSWIN prediction')
        return result
