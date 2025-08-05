import os
import json
import joblib
import pathlib
import gspread
import argparse
import numpy as np
import pandas as pd
import logging as log
from platformdirs import user_cache_dir
from google.oauth2 import service_account

from janome.tokenizer import Tokenizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

APP_NAME = "expense_type_classifier"
CACHE_PATH = pathlib.Path(user_cache_dir(APP_NAME))
CACHE_PATH.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()
HOME = os.getenv("HOME") or "~"
NA_TEXT = "No Memo"
log.basicConfig(
    level=LOG_LEVEL,
    handlers=[log.StreamHandler()],
    format="%(asctime)s - [%(levelname)s] %(message)s",
)


def main() -> None:
    log.info("start 'main' method")

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="Expense Type Prediction using Machine Learning"
    )
    parser.add_argument(
        "-j",
        "--json",
        dest="json_data",
        type=str,
        required=False,
        default=None,
        help='expense data in JSON format (required if --validate is not set) / example: \'{"memo": "Lunch at cafe", "amount": 1500}\'',
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
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="validate the trained model with the training data",
        default=False,
    )
    args = parser.parse_args()

    try:
        dim_reduction = args.dim_reduction
        if dim_reduction:
            args.predict_only = False
        if args.validate:
            validate_model(dim_reduction=dim_reduction)
            return
        if not args.predict_only:
            df = get_expense_history()
            df = preprocess_data(df)
            train(df_train=df, dim_reduction=dim_reduction)
        if not args.json_data:
            raise ValueError(
                "--json argument is required unless --validate is set"
            )
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
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("end 'main' method")


def get_expense_history() -> pd.DataFrame:
    log.info("start 'get_expense_history' method")
    expense_cache_path = pathlib.Path(user_cache_dir("expense"))
    fname = expense_cache_path / "expense_history.log"
    df = pd.read_csv(fname, index_col=None)
    df = df.T.reset_index().T
    df.columns = pd.Index(["date", "type", "memo", "amount"])
    df.index = pd.Index(range(len(df)))
    log.debug(f"Expense history DataFrame:\n{df.head()}")
    log.info("end 'get_expense_history' method")
    return df


def tokenize_text(text: str, tokenizer: Tokenizer) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return NA_TEXT
    processed_text = " ".join(list(tokenizer.tokenize(text, wakati=True)))
    return processed_text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("start 'preprocess_data' method")
    df_new = df.copy()
    # 日付をdatetime型に変換
    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
    # 金額を数値型に変換
    df_new["amount"] = (
        pd.to_numeric(df_new["amount"], errors="coerce").fillna(0).astype(int)
    )
    # メモを分かち書きにする
    tokenizer = Tokenizer()
    df_new["memo"] = df_new["memo"].apply(lambda s: tokenize_text(s, tokenizer))
    log.debug(f"Processed DataFrame:\n{df_new.head()}")
    log.info("end 'preprocess_data' method")
    return df_new


def train(df_train: pd.DataFrame, dim_reduction: bool = False) -> None:
    log.info("start 'train_and_save_model' method")
    df = df_train.copy()
    df = df.fillna(NA_TEXT)
    amount = df[["amount"]].astype(int).values
    # memoのテキストをTF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["memo"]).toarray()  # TF-IDFベクトル化
    log.debug(f"vectorizer features: {vectorizer.get_feature_names_out()}")
    if dim_reduction:
        dim_reducer = PCA(n_components=30)
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
    os.makedirs(CACHE_PATH, exist_ok=True)
    joblib.dump(clf, CACHE_PATH / "classifier.joblib")
    if dim_reduction:
        joblib.dump(dim_reducer, CACHE_PATH / "dim_reducer.joblib")
    joblib.dump(vectorizer, CACHE_PATH / "vectorizer.joblib")
    log.info("end 'train_and_save_model' method")


def predict(memo: str, amount: int, dim_reduction: bool = False) -> str:
    log.info("start 'predict' method")
    # モデルとvectorizerの読み込み
    clf = joblib.load(CACHE_PATH / "classifier.joblib")
    if dim_reduction:
        dim_reducer = joblib.load(CACHE_PATH / "dim_reducer.joblib")
    vectorizer = joblib.load(CACHE_PATH / "vectorizer.joblib")
    # メモを分かち書きにする
    memo = tokenize_text(memo, Tokenizer())
    # store_nameをベクトル化
    X_new = vectorizer.transform([memo]).toarray()  # TF-IDFベクトル化
    if dim_reduction:
        X_new = dim_reducer.transform(X_new)  # PCAで次元削減
    X_new = np.concatenate([X_new, [[amount]]], axis=1)  # ベクトルと金額を結合
    # カテゴリ予測
    predicted_type = str(clf.predict(X_new)[0])
    log.info("end 'predict' method")
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
    df.columns = pd.Index(
        list(df.columns[:-4]) + [f"memo{i+1}" for i in range(4)]
    )
    log.debug(f"DataFrame created: {df.head()}")
    df.to_csv("./train_data/spreadsheet_data.csv", index=True)
    log.info("end 'download_spreadsheet' method")


def validate_model(dim_reduction: bool = False) -> None:
    log.info("start 'validate_model' method")
    # トレーニングデータの取得と前処理
    df_org = get_expense_history()
    df = preprocess_data(df_org)
    train(df_train=df, dim_reduction=dim_reduction)
    # モデルとvectorizerの読み込み
    clf = joblib.load(CACHE_PATH / "classifier.joblib")
    if dim_reduction:
        dim_reducer = joblib.load(CACHE_PATH / "dim_reducer.joblib")
    vectorizer = joblib.load(CACHE_PATH / "vectorizer.joblib")
    # 特徴量の前処理
    df = df.fillna(NA_TEXT)
    amount = df[["amount"]].astype(int).values
    # memoのテキストをTF-IDFベクトル化
    X = vectorizer.transform(df["memo"]).toarray()  # TF-IDFベクトル化
    if dim_reduction:
        X = dim_reducer.transform(X)  # PCAで次元削減
    X = np.concatenate([X, amount], axis=1)  # ベクトルと金額を結合
    y = df["type"]  # 正解ラベル

    # 予測
    y_pred = clf.predict(X)
    df_result = df_org.copy()
    df_result["predicted_type"] = y_pred
    df_result["tokenized_memo"] = df["memo"]
    df_result = df_result[
        ["date", "amount", "memo", "tokenized_memo", "type", "predicted_type"]
    ]
    match = y_pred == y
    print(
        f"Validation accuracy: {np.mean(match)} ({np.sum(match)}/{len(match)})"
    )
    print(f"Validation results:\n{df_result}")
    log.info("end 'validate_model' method")


if __name__ == "__main__":
    main()
