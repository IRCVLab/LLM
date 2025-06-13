import numpy as np

def fit_parametric_poly(coords, num_samples=300):

    pts = np.asarray(coords, dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2:
        raise ValueError("coords must be list of >=2 (x,y) pairs")

    deltas  = np.diff(pts, axis=0)
    seg_len = np.sqrt((deltas**2).sum(axis=1))
    t       = np.concatenate([[0], np.cumsum(seg_len)])
    t      /= t[-1] 

    deg = len(t) - 1

    cx = np.polyfit(t, pts[:,0], deg)
    cy = np.polyfit(t, pts[:,1], deg)
    px = np.poly1d(cx)
    py = np.poly1d(cy)

    t_lin = np.linspace(0,1,num_samples)
    x_lin = px(t_lin)
    y_lin = py(t_lin)

    return x_lin, y_lin


def centripetal_catmull_rom(coords, num_samples=200, alpha=0.5):

    pts = np.asarray(coords, dtype=float)
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 points")

    deltas = np.diff(pts, axis=0)
    dist_alpha = np.sqrt((deltas**2).sum(axis=1))**alpha
    t_orig = [0.0]
    for da in dist_alpha:
        t_orig.append(t_orig[-1] + da)

    if len(t_orig) > 1:
        d0 = t_orig[1] - t_orig[0]
        dn = t_orig[-1] - t_orig[-2]
    else:
        d0 = dn = 1.0
    t = np.array([t_orig[0] - d0] + t_orig + [t_orig[-1] + dn], dtype=float)
    P = np.vstack([pts[0], pts, pts[-1]])

    t_min, t_max = t[1], t[-2]
    t_lin = np.linspace(t_min, t_max, num_samples)

    curve = []
    for u in t_lin:
        k = np.searchsorted(t, u) - 1
        k = np.clip(k, 1, n-1)

        P0, P1, P2, P3 = P[k-1], P[k], P[k+1], P[k+2]
        t0, t1, t2, t3 = t[k-1], t[k], t[k+1], t[k+2]

        A1 = (t1-u)/(t1-t0)*P0 + (u-t0)/(t1-t0)*P1
        A2 = (t2-u)/(t2-t1)*P1 + (u-t1)/(t2-t1)*P2
        A3 = (t3-u)/(t3-t2)*P2 + (u-t2)/(t3-t2)*P3

        B1 = (t2-u)/(t2-t0)*A1 + (u-t0)/(t2-t0)*A2
        B2 = (t3-u)/(t3-t1)*A2 + (u-t1)/(t3-t1)*A3

        C  = (t2-u)/(t2-t1)*B1 + (u-t1)/(t2-t1)*B2

        curve.append(C)

    curve = np.asarray(curve)
    return curve[:,0], curve[:,1]