"""Constant-imputed, standardized Ridge from chunked sufficient statistics.

Equivalent objective to SimpleImputer(0, add_indicator=True), StandardScaler,
Ridge(fit_intercept=True); avoid duplicating daily solar vectors a million times.
"""
import numpy as np
from scipy.linalg import solve


def design(frame, solar, columns):
    indices = frame.aia_frame_index.fillna(0).to_numpy(dtype=int)
    values = np.empty((len(frame),len(columns)),dtype=np.float64)
    for j,name in enumerate(columns):
        values[:,j] = frame[name].to_numpy() if name in frame else solar[name].to_numpy()[indices]
    return values


def fit(frame, solar, columns, alpha=100., chunk_size=4096):
    missing = np.zeros(len(columns),dtype=bool)
    for begin in range(0,len(frame),chunk_size):
        x=design(frame.iloc[begin:begin+chunk_size],solar,columns)
        if np.isinf(x).any(): raise ValueError("Infinite feature")
        missing |= np.isnan(x).any(axis=0)
    indicators=np.flatnonzero(missing)
    size=len(columns)+len(indicators)
    gram=np.zeros((size,size)); total=np.zeros(size); cross=np.zeros(size)
    ysum=0.; count=0
    for begin in range(0,len(frame),chunk_size):
        part=frame.iloc[begin:begin+chunk_size]
        x=design(part,solar,columns)
        x=np.column_stack([np.nan_to_num(x,nan=0.),np.isnan(x[:,indicators]).astype(float)])
        y=(part.target_v-part.dlinear_v).to_numpy(dtype=float)
        if not np.isfinite(y).all(): raise ValueError("Unobserved training targets")
        gram+=x.T@x; total+=x.sum(axis=0); cross+=x.T@y; ysum+=y.sum(); count+=len(y)
    if not count: raise ValueError("Empty training split")
    mean=total/count; ymean=ysum/count
    centered=gram-np.outer(total,mean)
    var=np.maximum(np.diag(centered)/count,0.)
    scale=np.sqrt(var)
    scale[var <= (10*np.finfo(float).eps*np.maximum(1,mean**2))]=1.
    matrix=centered/scale[:,None]/scale[None,:]
    matrix.flat[::size+1]+=alpha
    coefficient=solve(matrix,(cross-total*ymean)/scale,assume_a="pos")/scale
    return dict(columns=list(columns),indicators=indicators,coefficient=coefficient,
                intercept=float(ymean-mean@coefficient),alpha=alpha,training_rows=count)


def predict(frame,solar,bundle,chunk_size=4096):
    output=np.empty(len(frame))
    for begin in range(0,len(frame),chunk_size):
        x=design(frame.iloc[begin:begin+chunk_size],solar,bundle["columns"])
        x=np.column_stack([np.nan_to_num(x,nan=0.),np.isnan(x[:,bundle["indicators"]]).astype(float)])
        output[begin:begin+len(x)]=x@bundle["coefficient"]+bundle["intercept"]
    return output
