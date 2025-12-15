from betta_pipeline import helper_functions_VA
import numpy as np        # for array math and trigonometric functions
import pandas as pd       # if data_auto is a DataFrame
import glob
import os

# Define input and output paths


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def midpoint(pos_1, pos_2, pos_3, pos_4):
    """Compute midpoint between two points."""
    midpointx = (pos_1 + pos_3) / 2
    midpointy = (pos_2 + pos_4) / 2
    return (midpointx, midpointy)


def mydistance(pos_1, pos_2):
    """Compute distance between two (x, y) coordinates."""
    x0, y0 = pos_1
    x1, y1 = pos_2
    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return dist


def coords(data):
    """Helper for readability."""
    return (data['x'], data['y'])


def lawofcosines(line_1, line_2, line_3):
    """Compute angle between line_1 and line_2 using law of cosines."""
    num = line_1 ** 2 + line_2 ** 2 - line_3 ** 2
    denom = (line_1 * line_2) * 2
    floatfraction = num.astype(float) / denom.astype(float)
    OPdeg = vecnanarccos()(floatfraction)
    return OPdeg


def vecnanarccos():
    A = np.frompyfunc(nanarccos, 1, 1)
    return A


def nanarccos(floatfraction):
    con1 = ~np.isnan(floatfraction)
    if con1:
        floatfraction = np.clip(floatfraction, -1, 1)
        cos = np.arccos(floatfraction)
        OPdeg = np.degrees(cos)
    else:
        OPdeg = np.nan
    return OPdeg


def operculum(df):
    poi = ['A_head', 'B_rightoperculum', 'E_leftoperculum']
    HROP = mydistance(coords(df[poi[0]]), coords(df[poi[1]]))
    HLOP = mydistance(coords(df[poi[0]]), coords(df[poi[2]]))
    RLOP = mydistance(coords(df[poi[1]]), coords(df[poi[2]]))
    Operangle = lawofcosines(HROP, HLOP, RLOP)
    return Operangle


def rename_df(df):
    """Rename DLC bodyparts for consistent naming."""
    try:
        new_bodyparts = {
            "head": "A_head",
            "lefteye": "M_lefteye",
            "righteye": "N_righteye",
            "leftoperculum": "E_leftoperculum",
            "rightoperculum": "B_rightoperculum",
            "spine1": "F_spine1",
            "spine2": "G_spine2",
            "spine3": "H_spine3",
            "spine4": "I_spine4",
            "tailbase": "C_tailbase",
            "tailtip": "D_tailtip"
        }
        df = df.rename(columns=new_bodyparts, level=1)
        df.columns = df.columns.droplevel(0)
        return df
    except Exception:
        df.columns = df.columns.droplevel(0)
        return df


def orientation(data_auto):
    """Compute orientation relative to x-axis."""
    head_x = data_auto["A_head"]["x"]
    head_y = data_auto["A_head"]["y"]
    mid_x = midpoint(
        data_auto['B_rightoperculum']['x'],
        data_auto['B_rightoperculum']['y'],
        data_auto['E_leftoperculum']['x'],
        data_auto['E_leftoperculum']['y']
    )[0]
    mid_y = midpoint(
        data_auto['B_rightoperculum']['x'],
        data_auto['B_rightoperculum']['y'],
        data_auto['E_leftoperculum']['x'],
        data_auto['E_leftoperculum']['y']
    )[1]
    head_ori = np.array([head_x - mid_x, head_y - mid_y]).T
    ref = np.array([1, 0])
    inner_product = head_ori.dot(ref)
    cos = inner_product / np.sqrt(np.sum(np.multiply(head_ori, head_ori), axis=1))
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos) / np.pi * 180
    return angle


def speed(data, fps=40):
    """Compute instantaneous speed based on head motion."""
    poi = ['A_head']
    (Xcoords, Ycoords) = coords(data[poi[0]])
    Xcoords = Xcoords.rolling(window=3, center=True, min_periods=1).mean()
    Ycoords = Ycoords.rolling(window=3, center=True, min_periods=1).mean()
    distx = Xcoords.diff()
    disty = Ycoords.diff()
    TotalDist = np.sqrt(distx ** 2 + disty ** 2)
    Speed = TotalDist / (1 / fps)
    Speed[0:3] = Speed[3]
    return Speed


def turning_angle_spine(data_auto):
    """Compute turning angle of the spine across 1 second."""
    spine1_x = data_auto["F_spine1"]["x"]
    spine1_y = data_auto["F_spine1"]["y"]
    if "mid_spine1_spine2" in data_auto.columns:
        spine1_5_x = data_auto["mid_spine1_spine2"]["x"]
        spine1_5_y = data_auto["mid_spine1_spine2"]["y"]
    else:
        spine1_5_x = data_auto["G_spine2"]["x"]
        spine1_5_y = data_auto["G_spine2"]["y"]
    cur_vec = np.vstack((spine1_x - spine1_5_x, spine1_y - spine1_5_y)).T
    prev_vec = np.vstack((
        np.append(np.repeat(np.nan, 40), (spine1_x - spine1_5_x)[:-40]),
        np.append(np.repeat(np.nan, 40), (spine1_y - spine1_5_y)[:-40])
    )).T
    inner_product = np.sum(np.multiply(cur_vec, prev_vec), axis=1)
    cur_norm = np.sum(np.multiply(cur_vec, cur_vec), axis=1)
    prev_norm = np.sum(np.multiply(prev_vec, prev_vec), axis=1)
    cos = inner_product / np.sqrt(cur_norm * prev_norm)
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)
    angle[0:40] = angle[40]
    return angle / np.pi * 180


def tail_angle(df):
    """Compute tail bending angle relative to body axis."""
    p3 = df['H_spine3'][['x', 'y']].to_numpy()
    p4 = df['I_spine4'][['x', 'y']].to_numpy()
    pb = df['C_tailbase'][['x', 'y']].to_numpy()
    pt = df['D_tailtip'][['x', 'y']].to_numpy()
    pts_body = np.stack([p3, p4, pb], axis=1)
    body_vec = np.zeros((len(df), 2))
    for i in range(len(df)):
        x = pts_body[i, :, 0]
        y = pts_body[i, :, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        body_vec[i] = np.array([1, m]) / np.linalg.norm([1, m])
    tail_vec = pt - pb
    tail_vec = tail_vec / np.linalg.norm(tail_vec, axis=1, keepdims=True)
    dot = np.sum(body_vec * tail_vec, axis=1)
    dot = np.clip(dot, -1, 1)
    tail_angle = np.degrees(np.arccos(dot))
    return tail_angle


def tail_dev(df):
    """Compute perpendicular deviation of tailtip from body axis."""
    p3 = df['H_spine3'][['x', 'y']].to_numpy()
    p4 = df['I_spine4'][['x', 'y']].to_numpy()
    pb = df['C_tailbase'][['x', 'y']].to_numpy()
    pt = df['D_tailtip'][['x', 'y']].to_numpy()
    pts_body = np.stack([p3, p4, pb], axis=1)
    m = np.zeros(len(df))
    c = np.zeros(len(df))
    for i in range(len(df)):
        x = pts_body[i, :, 0]
        y = pts_body[i, :, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m[i], c[i] = np.linalg.lstsq(A, y, rcond=None)[0]
    u = np.stack([np.ones_like(m), m], axis=1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    anchor = pb
    v = pt - anchor
    signed_dev = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    tail_dev = np.abs(signed_dev)
    return tail_dev


def auto_scoring_get_angle(data_auto, poi_1):
    """Compute flare angle for one side using law of cosines."""
    HROP_1 = mydistance(coords(data_auto[poi_1[0]]), coords(data_auto[poi_1[1]]))
    HLOP_1 = mydistance(coords(data_auto[poi_1[0]]), coords(data_auto[poi_1[2]]))
    RLOP_1 = mydistance(coords(data_auto[poi_1[1]]), coords(data_auto[poi_1[2]]))
    Operangle_1 = lawofcosines(HROP_1, HLOP_1, RLOP_1)
    return Operangle_1


def oper_diff(data_auto, poi_1):
    """Compute distance between operculum and reference point."""
    dist = mydistance(coords(data_auto[poi_1[0]]), coords(data_auto[poi_1[1]]))
    return dist


# ------------------------------------------------------
# Main processing function
# ------------------------------------------------------
def concating(df, filename):
    print(filename)
    output = pd.DataFrame()
    output['operculum'] = operculum(df)

    if "L" in filename:
        print("yes")
        orient = orientation(df)
        orient = 180 - orient
        head_x = 500 - df["A_head"]["x"]
        tail_x = 500 - df['D_tailtip']['x']
    else:
        orient = orientation(df)
        head_x = df["A_head"]["x"]
        tail_x = df['D_tailtip']['x']

    output['orientation'] = orient
    print(orient[0])
    output['movement_speed'] = speed(df)
    output['turning_angle'] = turning_angle_spine(df)
    output["head_x"] = head_x
    output['head_y'] = df['A_head']['y']
    output['centroid_x'] = (df['G_spine2']['x'] + df['H_spine3']['x'] + df['I_spine4']['x']) / 3
    output['centroid_y'] = (df['G_spine2']['y'] + df['H_spine3']['y'] + df['I_spine4']['y']) / 3
    output['tail_x'] = tail_x
    output['tail_y'] = df['D_tailtip']['y']
    output['tail_angle'] = tail_angle(df)
    output['tail_dev'] = tail_dev(df)

    operangle_R = auto_scoring_get_angle(df, ['A_head', 'B_rightoperculum', 'F_spine1'])
    operangle_L = auto_scoring_get_angle(df, ['A_head', 'E_leftoperculum', 'F_spine1'])
    operdist_R = oper_diff(df, ['B_rightoperculum', 'F_spine1'])
    operdist_L = oper_diff(df, ['E_leftoperculum', 'F_spine1'])

    output['oper_angle_R'] = operangle_R
    output['oper_angle_L'] = operangle_L
    output['oper_dist_R'] = operdist_R
    output['oper_dist_L'] = operdist_L
    output['oper_angle_avg'] = (operangle_R + operangle_L) / 2
    output['oper_dist_avg'] = (operdist_R + operdist_L) / 2
    return output


# ------------------------------------------------------
# Batch processing
# ------------------------------------------------------
def feature_generation(files,output_path):
    for file in files:
        filename = os.path.basename(file)
        print(f"\nProcessing: {filename}")

        # 1. Load depending on file extension
        if file.endswith(".h5"):
            print("📂 Detected HDF5 DeepLabCut file")
            df = pd.read_hdf(file)
            df = df.droplevel(0, axis=1)
            if not isinstance(df.columns, pd.MultiIndex):
                print("⚠️ Unexpected format — columns are not MultiIndex. Check DLC export.")
        elif file.endswith(".csv"):
            print("📄 Detected CSV file")
            df = pd.read_csv(file, header=[0, 1, 2])
            df = rename_df(df)
        else:
            print(f"⚠️ Skipping unsupported file type: {filename}")
            continue

        # 2. Generate features
        output = concating(df, filename)

        # 3. Save output
        outname = os.path.splitext(filename)[0] + ".csv"
        output.to_csv(os.path.join(output_path, outname), index=True)
        print(f"✅ Output saved: {outname}")
