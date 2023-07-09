import argparse
import json
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    args = parser.parse_args()

    # read the JSON file as produced by PyTesseract
    ocr_data = json.load(open(args.json))
    sort_order = "page_num block_num par_num line_num word_num".split()
    df_ocr = pd.DataFrame(ocr_data).sort_values(by=sort_order)
    df_ocr = df_ocr[df_ocr["text"] != ""]
    df_ocr["alnum"] = df_ocr["text"].str.isalnum()
    df_ocr["len"] = df_ocr["text"].str.len()
    # Get statistics: mean, median, 90%-ile, 95%-ile, max
    mini = df_ocr["conf"].min()
    maxi = df_ocr["conf"].max()
    mean = df_ocr["conf"].mean()
    medi = df_ocr["conf"].median()
    pc95 = df_ocr["conf"].quantile(0.05)
    pc99 = df_ocr["conf"].quantile(0.01)
    print(mini, pc99, pc95, mean, medi, maxi)
    # again: weighted quantile
    #   weighted by number of letters/chars
    order = df_ocr["len"].iloc[df_ocr["conf"].argsort()].cumsum()
    order = order / order.iloc[-1]
    quantiles = [0, 0.01, 0.05, 0.5, 1]
    bin = pd.cut(order, quantiles)  # not quantile value, but match value into bin
    df_weighted = pd.DataFrame({'bin':bin, 'order':order}).join(df_ocr)
    df_weighted['prev_bin'] = df_weighted['bin'].shift(1)
    df_weighted['prev_order'] = df_weighted['order'].shift(1)
    df_weighted['prev_conf'] = df_weighted['conf'].shift(1)
    wpmean = (df_weighted['conf']*df_weighted['len']).sum()/df_weighted['len'].sum()
    wp99 = wp95 = wp50 = None
    for rec in df_weighted[df_weighted['bin'].ne(df_weighted['prev_bin'])].dropna().filter(['bin','prev_bin','order','prev_order','conf','prev_conf']).to_dict('records'):
        x1, x2, y1, y2 = rec['prev_order'], rec['order'], rec['prev_conf'], rec['conf']
        if x1 <= 0.01 <= x2:
            wp99 = y1 + (y2-y1) * (0.01-x1) / (x2-x1)
        if x1 <= 0.05 <= x2:
            wp95 = y1 + (y2-y1) * (0.05-x1) / (x2-x1)
        if x1 <= 0.50 <= x2:
            wp50 = y1 + (y2-y1) * (0.50-x1) / (x2-x1)
    nwords = len(df_ocr)
    nchars = df_ocr["len"].sum()
    print(wp99, wp95, wp50, wpmean, nwords, nchars)
    # again: only the alnum strings because symbols are known weakness
    df = df_ocr[df_ocr["alnum"]]
    mini = df["conf"].min()
    maxi = df["conf"].max()
    mean = df["conf"].mean()
    medi = df["conf"].median()
    pc95 = df["conf"].quantile(0.05)
    pc99 = df["conf"].quantile(0.01)
    print(mini, pc99, pc95, mean, medi, maxi)
