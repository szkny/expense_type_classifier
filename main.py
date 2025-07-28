import os
import json
import joblib
import gspread
import argparse
import numpy as np
import pandas as pd
import logging as log
from google.oauth2 import service_account

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()
HOME = os.getenv("HOME") or "~"
NA_TEXT = "No Memo"
log.basicConfig(
    level=LOG_LEVEL,
    handlers=[log.StreamHandler()],
    format="%(asctime)s - [%(levelname)s] %(message)s",
)


def main(args: argparse.Namespace) -> None:
    log.info("start 'main' method")
    # # download_spreadsheet()
    dim_reduction = args.dim_reduction
    if dim_reduction:
        args.re_train = True
    df = get_expense_history()
    if not args.predict_only:
        train(df_train=df, dim_reduction=dim_reduction)
    log.debug(f"--json: {args.json_data}")
    input_data = json.loads(args.json_data)
    log.debug(f"loaded json data: {input_data}")
    predicted_type = predict(
        memo=input_data.get("memo", NA_TEXT),
        amount=input_data.get("amount", 0),
        dim_reduction=dim_reduction,
    )
    log.debug(f"Predicted type: {predicted_type}")
    # 予測結果をJSON形式で出力
    output_data = {
        "memo": input_data.get("memo", NA_TEXT),
        "amount": input_data.get("amount", 0),
        "predicted_type": predicted_type,
    }
    result = json.dumps(output_data, ensure_ascii=False, indent=2)
    print(result, end="")
    log.info("end 'main' method")


def get_expense_history() -> pd.DataFrame:
    df = pd.read_csv(HOME + "/tmp/expense/expense_history.log", index_col=None)
    df = df.T.reset_index().T
    df.columns = pd.Index(["date", "type", "memo", "amount"])
    df.index = pd.Index(range(len(df)))
    return df


def train(df_train: pd.DataFrame, dim_reduction=False) -> None:
    log.info("start 'train_and_save_model' method")
    df = df_train.copy()
    df = df.fillna(NA_TEXT)
    amount = df[["amount"]].astype(int).values
    # store_nameのテキストをTF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    dim_reducer = PCA(n_components=10)
    X = vectorizer.fit_transform(df["memo"]).toarray()  # TF-IDFベクトル化
    if dim_reduction:
        X = dim_reducer.fit_transform(X)  # PCAで次元削減
    X = np.concatenate([X, amount], axis=1)  # ベクトルと金額を結合
    y = df["type"]  # 正解ラベル
    log.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    log.debug(f"X:\n{X}")

    # 分類器の作成・学習
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    log.info("Model trained successfully")

    # 保存
    joblib.dump(clf, "cache/classifier.joblib")
    if dim_reduction:
        joblib.dump(dim_reducer, "cache/dim_reducer.joblib")
    joblib.dump(vectorizer, "cache/vectorizer.joblib")
    log.info("end 'train_and_save_model' method")


def predict(memo: str, amount: int, dim_reduction=False) -> str:
    # モデルとベクトルライザの読み込み
    clf = joblib.load("cache/classifier.joblib")
    if dim_reduction:
        dim_reducer = joblib.load("cache/dim_reducer.joblib")
    vectorizer = joblib.load("cache/vectorizer.joblib")
    if not memo or (type(memo) is float and np.isnan(memo)):
        memo = NA_TEXT
    # store_nameをベクトル化
    X_new = vectorizer.transform([memo]).toarray()  # TF-IDFベクトル化
    if dim_reduction:
        X_new = dim_reducer.transform(X_new)  # PCAで次元削減
    X_new = np.concatenate([X_new, [[amount]]], axis=1)  # ベクトルと金額を結合
    # カテゴリ予測
    predicted_type = clf.predict(X_new)[0]
    return predicted_type


def download_spreadsheet() -> None:
    log.info("start 'download_spreadsheet' method")
    credentials = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(credentials)

    # スプレッドシートのダウンロード
    sheetname_list: list[str] = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    workbook_list = [
        "CF (2020年度)",
        "CF (2021年度)",
        "CF (2022年度)",
        "CF (2023年度)",
        "CF (2024年度)",
        "CF (2025年度)",
    ]
    workbook = client.open(workbook_list[4])
    sheetname = sheetname_list[0]
    worksheet = workbook.worksheet(sheetname)
    data = worksheet.get_all_records(head=30)
    log.debug(f"Data downloaded from {sheetname} sheet: {data}")

    # TODO: 値ではなく関数を取得する
    # # d = worksheet.acell("B31:AG43", value_render_option="FORMULA")
    # cells = worksheet.get("B31:AG43", value_render_option="FORMULA")
    # log.debug(f"d: {cells}")
    # # d = [_d.value for _d in cells]
    # # log.debug(f"d: {d}")

    # TODO: 支出タイプ・メモ・金額で１サンプルのデータフレームを作成する

    # データをCSVに保存
    df = pd.DataFrame(data).head(24).set_index("日付").T
    df.index.name = "date"
    df.columns = pd.Index(list(df.columns[:-4]) + [f"memo{i+1}" for i in range(4)])
    log.debug(f"DataFrame created: {df.head()}")
    df.to_csv("./train_data/spreadsheet_data.csv", index=True)
    log.info("end 'download_spreadsheet' method")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="家計簿スプレッドシートに自動で書き込みを行うバッチプログラム"
    )
    parser.add_argument(
        "-j",
        "--json",
        dest="json_data",
        type=str,
        required=True,
        default=None,
        help="expense data in JSON format",
    )
    parser.add_argument(
        "-p",
        "--predict-only",
        dest="predict_only",
        action="store_true",
        help="predict without re-train the model with the latest expense history",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--dim-reduction",
        dest="dim_reduction",
        action="store_true",
        help="enable PCA for dimensionality reduction",
        default=False,
    )
    args = parser.parse_args()
    main(args)
