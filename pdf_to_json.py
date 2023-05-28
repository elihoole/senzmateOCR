from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
from glob import glob
import json
import re


class PDFToOCR:
    def __init__(
        self,
        pdf_path=None,
        from_json=False,
        ocr_json_path=None,
        page_num=2,
    ):
        self.pdf_path = pdf_path
        if not from_json:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True, lang="en", page_num=page_num
            )

            self.ocr_results = self.ocr_engine.ocr(self.pdf_path, cls=True)
        else:
            self.ocr_engine = None
            self.ocr_results = self.load_ocr_results_from_json(ocr_json_path)

    def perform_ocr(self):
        self.ocr_results = self.ocr_engine.ocr(self.pdf_path, cls=True)

    def save_ocr_results_as_json(self):
        with open("ocr_results.json", "w") as f:
            json.dump({"results": self.ocr_results}, f)

    def load_ocr_results_from_json(self, ocr_json_path):
        with open(ocr_json_path, "r") as f:
            return json.load(f)["results"]

    def save_page_df_as_csv(self):
        for i, page in enumerate(self.ocr_results):
            page_df = pd.DataFrame(page)
            page_df.to_csv(f"page_{i}.csv", index=False)


class SingGenHospInvoice:
    def __init__(self):
        self.hospital_names = ["Singapore General Hospital", "Tan Tock Seng Hospital"]
        self.invoice_df = None
        self.hospital_name = None
        self.key_info = None
        self.invoice_table = None
        self.margin = 5
        self.invoice_key_info_fields = [
            "Tax Invoice Number",
            "Bill Ref Number",
            "Tax Invoice Date",
            "Patient NRICI/HRN",
            "Visit Date",
            "Visit/Bill Location",
            "Payment Class",
            "Type of Supply",
            "GST Reg No",
            "Page No",
            "Bill Type",
        ]
        self.gst_to_page_info = None
        self.invoice_json = None
        self.total_payments_info = None

    def make_invoice_df(self, page):
        df = pd.DataFrame(page, columns=["bbox", "text"])
        df["text"] = df["text"].apply(lambda x: x[0].strip())
        df["bbox_upper_left_x"] = df["bbox"].apply(lambda x: x[0][0])
        df["bbox_upper_left_y"] = df["bbox"].apply(lambda x: x[0][1])
        df["bbox_upper_right_x"] = df["bbox"].apply(lambda x: x[1][0])
        df["bbox_upper_right_y"] = df["bbox"].apply(lambda x: x[1][1])
        df["bbox_lower_right_x"] = df["bbox"].apply(lambda x: x[2][0])
        df["bbox_lower_right_y"] = df["bbox"].apply(lambda x: x[2][1])
        df["bbox_lower_left_x"] = df["bbox"].apply(lambda x: x[3][0])
        df["bbox_lower_left_y"] = df["bbox"].apply(lambda x: x[3][1])
        self.invoice_df = df

    def get_hospital_name_bbox(self):
        upper_left_x = (
            min(
                self.invoice_df.iloc[0]["bbox_upper_left_x"],
                self.invoice_df[
                    self.invoice_df["text"].str.contains("hospital", case=False)
                ].iloc[0]["bbox_upper_left_x"],
            )
            - self.margin
        )

        upper_left_y = (
            min(
                self.invoice_df.iloc[0]["bbox_upper_left_y"],
                self.invoice_df[
                    self.invoice_df["text"].str.contains("hospital", case=False)
                ].iloc[0]["bbox_lower_right_y"],
            )
            - self.margin
        )

        lower_right_x = (
            max(
                self.invoice_df.iloc[0]["bbox_lower_right_x"],
                self.invoice_df[
                    self.invoice_df["text"].str.contains("hospital", case=False)
                ].iloc[0]["bbox_lower_right_x"],
            )
            + self.margin
        )
        lower_right_y = (
            max(
                self.invoice_df.iloc[0]["bbox_lower_right_y"],
                self.invoice_df[
                    self.invoice_df["text"].str.contains("hospital", case=False)
                ].iloc[0]["bbox_lower_right_y"],
            )
            + self.margin
        )
        return upper_left_x, upper_left_y, lower_right_x, lower_right_y

    def get_hospital_name(self):
        (
            upper_left_x,
            upper_left_y,
            lower_right_x,
            lower_right_y,
        ) = self.get_hospital_name_bbox()
        hospital_name = (
            " ".join(
                self.invoice_df[
                    (self.invoice_df["bbox_upper_left_x"] > upper_left_x)
                    & (self.invoice_df["bbox_upper_left_y"] > upper_left_y)
                    & (self.invoice_df["bbox_lower_right_x"] < lower_right_x)
                    & (self.invoice_df["bbox_lower_right_y"] < lower_right_y)
                ]["text"].to_list()
            )
            .strip()
            .title()
        )

        if hospital_name not in self.hospital_names:
            raise ValueError("Hospital name not found")
        else:
            self.hospital_name = hospital_name

    def get_gst_to_page_number_bbox(self):
        upper_left_x = self.invoice_df["bbox_upper_left_x"].min() - self.margin
        upper_left_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("TAX INVOICE")][
                "bbox_lower_left_y"
            ].values[0]
            + self.margin
        )
        lower_right_x = self.invoice_df["bbox_lower_right_x"].max() + self.margin
        lower_right_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("Tax Invoice Number")][
                "bbox_upper_left_y"
            ].values[0]
            + self.margin
        )

        return upper_left_x, upper_left_y, lower_right_x, lower_right_y

    def get_gst_to_page_number_df(self):
        (
            upper_left_x,
            upper_left_y,
            lower_right_x,
            lower_right_y,
        ) = self.get_gst_to_page_number_bbox()
        gst_to_page_number_df = self.invoice_df[
            (self.invoice_df["bbox_upper_left_x"] > upper_left_x)
            & (self.invoice_df["bbox_upper_left_y"] > upper_left_y)
            & (self.invoice_df["bbox_lower_right_x"] < lower_right_x)
            & (self.invoice_df["bbox_lower_right_y"] < lower_right_y)
        ]
        return gst_to_page_number_df.reset_index(drop=True)

    def get_gst_to_page_number_info(self):
        gst_to_page_number_df = self.get_gst_to_page_number_df()
        gst_to_page_num_dict = {}
        gst_to_page_num_dict["gst_number"] = (
            gst_to_page_number_df[
                gst_to_page_number_df["text"].str.contains("GST REG NO")
            ]["text"]
            .values[0]
            .split(":")[-1]
            .strip()
        )
        gst_to_page_num_dict["bill_type"] = gst_to_page_number_df[
            gst_to_page_number_df["text"].str.contains("ORIGINAL|DUPLICATE|INTERIM")
        ]["text"].values[0]
        gst_to_page_num_dict["page_number"] = (
            gst_to_page_number_df[gst_to_page_number_df["text"].str.contains("Page")][
                "text"
            ]
            .values[0]
            .split("/")[-1]
            .replace("Page", "")
            .strip()
        )
        tax_date = re.findall(
            r"\d{2}[\.\s]+\d{2}[\.\s]+\d{4}",
            gst_to_page_number_df[gst_to_page_number_df["text"].str.contains("Page")][
                "text"
            ].values[0],
        )
        if len(tax_date) > 0:
            gst_to_page_num_dict["tax_invoice_date"] = tax_date[0]
        else:
            gst_to_page_num_dict["tax_invoice_date"] = ""
        self.gst_to_page_info = gst_to_page_num_dict

    def get_key_info_bbox(self):
        upper_left_x = (
            self.invoice_df[self.invoice_df["text"].str.contains("Tax Invoice Number")][
                "bbox_upper_left_x"
            ].values[0]
            - self.margin
        )
        upper_left_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("Tax Invoice Number")][
                "bbox_upper_left_y"
            ].values[0]
            - self.margin
        )

        # lower right x coordinate is the max of the x coordinates in the sub dataframe between 'Tax Invoice Number' and 'Type of Supply'

        index_of_tax_invoice_number = self.invoice_df[
            self.invoice_df["text"].str.contains("Tax Invoice Number")
        ].index[0]
        index_of_type_of_supply = self.invoice_df[
            self.invoice_df["text"].str.contains("Type of Supply")
        ].index[0]

        lower_right_x = (
            self.invoice_df.iloc[index_of_tax_invoice_number:index_of_type_of_supply][
                "bbox_lower_right_x"
            ].max()
            + self.margin
        )

        lower_right_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("Type of Supply")][
                "bbox_lower_right_y"
            ].values[0]
            + self.margin
        )

        return upper_left_x, upper_left_y, lower_right_x, lower_right_y

    def get_key_info_box_df(self):
        (
            upper_left_x,
            upper_left_y,
            lower_right_x,
            lower_right_y,
        ) = self.get_key_info_bbox()

        df_key_info_box = self.invoice_df[
            (self.invoice_df["bbox_upper_left_x"] > upper_left_x - self.margin)
            & (self.invoice_df["bbox_upper_left_y"] > upper_left_y - self.margin)
            & (self.invoice_df["bbox_lower_right_x"] < lower_right_x + self.margin)
            & (self.invoice_df["bbox_lower_right_y"] < lower_right_y + self.margin)
        ].copy()

        return df_key_info_box.reset_index(drop=True)

    def get_key_info(self):
        df_key_info_box = self.get_key_info_box_df()
        df_key_info_box = self.check_if_same_line_append_newline_character(
            df_key_info_box
        )
        key_info_box_text = " ".join(df_key_info_box["text_with_newline"].values)
        key_info_box_text_corrected = self.replace_visit_location(key_info_box_text)

        key_info_box_dict = {}
        for line in key_info_box_text_corrected.split("\n"):
            line = line.strip()
            if line.find(":") != -1:
                key_info_box_dict[line.split(":")[0].strip()] = line.split(":")[
                    1
                ].strip()
            else:
                field_of_line = self.check_if_line_starts_with_field_name(line)
                if field_of_line:
                    key_info_box_dict[field_of_line] = line.replace(
                        field_of_line, ""
                    ).strip()
        self.key_info = {
            k.title().replace(" ", ""): v for k, v in key_info_box_dict.items()
        }

    def get_invoice_table_bbox(self):
        upper_left_x = (
            self.invoice_df[self.invoice_df["text"].str.contains("SERVICE CODE")][
                "bbox_upper_left_x"
            ].values[0]
            - self.margin
        )
        upper_left_y = (
            min(
                self.invoice_df[self.invoice_df["text"].str.contains("SERVICE CODE")][
                    "bbox_upper_left_y"
                ].values[0],
                self.invoice_df[self.invoice_df["text"].str.contains("AMOUNT")][
                    "bbox_upper_right_y"
                ].values[0],
            )
            - self.margin
        )

        lower_right_x = self.invoice_df["bbox_upper_right_x"].max() + self.margin
        lower_right_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("Subtotal Charges")][
                "bbox_lower_right_y"
            ].values[0]
            + self.margin
        )

        return upper_left_x, upper_left_y, lower_right_x, lower_right_y

    def get_invoice_table_df(self):
        (
            upper_left_x,
            upper_left_y,
            lower_right_x,
            lower_right_y,
        ) = self.get_invoice_table_bbox()

        df_invoice_table = self.invoice_df[
            (self.invoice_df["bbox_upper_left_x"] > upper_left_x)
            & (self.invoice_df["bbox_upper_left_y"] > upper_left_y)
            & (self.invoice_df["bbox_lower_right_x"] < lower_right_x)
            & (self.invoice_df["bbox_lower_right_y"] < lower_right_y)
        ].copy()

        return df_invoice_table.reset_index(drop=True)

    def get_invoice_table_by_column(self, df_invoice_table):
        (
            description_border_left_x,
            quantity_border_left_x,
            amount_border_left_x,
        ) = self.get_invoice_table_column_borders(df_invoice_table)

        df_invoice_table_codes = df_invoice_table[
            df_invoice_table["bbox_upper_right_x"] < description_border_left_x
        ].copy()

        df_invoice_table_minus_codes = df_invoice_table[
            df_invoice_table.index.isin(df_invoice_table_codes.index) == False
        ].copy()

        df_invoice_table_description = df_invoice_table_minus_codes[
            (
                df_invoice_table_minus_codes["bbox_upper_right_x"]
                < quantity_border_left_x
            )
        ].copy()

        df_invoice_table_minus_codes_and_description = df_invoice_table_minus_codes[
            df_invoice_table_minus_codes.index.isin(df_invoice_table_description.index)
            == False
        ].copy()

        df_invoice_table_quantity = df_invoice_table_minus_codes_and_description[
            (
                df_invoice_table_minus_codes_and_description["bbox_upper_right_x"]
                < amount_border_left_x
            )
        ].copy()

        df_invoice_table_amount = df_invoice_table_minus_codes_and_description[
            df_invoice_table_minus_codes_and_description.index.isin(
                df_invoice_table_quantity.index
            )
            == False
        ].copy()

        return self.clean_invoice_table_columns(
            (
                df_invoice_table_codes,
                df_invoice_table_description,
                df_invoice_table_quantity,
                df_invoice_table_amount,
            )
        )

    def align_invoice_table_columns(self):
        (
            df_invoice_table_codes,
            df_invoice_table_description,
            df_invoice_table_quantity,
            df_invoice_table_amount,
        ) = self.get_invoice_table_by_column(self.get_invoice_table_df())

        line_items = []
        for i, row in df_invoice_table_codes.iterrows():
            line_items.append(
                {
                    "item_code": row["text"],
                    "item_description": df_invoice_table_description.iloc[
                        (
                            df_invoice_table_description["bbox_upper_left_y"]
                            - row["bbox_upper_left_y"]
                        )
                        .abs()
                        .argsort()[:1]
                    ]["text"].values[0],
                    "item_quantity": df_invoice_table_quantity.iloc[
                        (
                            df_invoice_table_quantity["bbox_upper_left_y"]
                            - row["bbox_upper_left_y"]
                        )
                        .abs()
                        .argsort()[:1]
                    ]["text"].values[0],
                    "item_amount": df_invoice_table_amount.iloc[
                        (
                            df_invoice_table_amount["bbox_upper_left_y"]
                            - row["bbox_upper_left_y"]
                        )
                        .abs()
                        .argsort()[:1]
                    ]["text"].values[0],
                }
            )

        self.invoice_table = line_items

    def get_invoice_table_column_borders(self, df_invoice_table):
        description_border_left_x = (
            df_invoice_table[df_invoice_table["text"].str.contains("DESCRIPTION")][
                "bbox_upper_left_x"
            ].min()
            - self.margin
        )
        quantity_border_left_x = (
            df_invoice_table[df_invoice_table["text"].str.contains("QUANTITY")][
                "bbox_upper_left_x"
            ].min()
            - self.margin
        )
        amount_border_left_x = (
            df_invoice_table[df_invoice_table["text"].str.contains("Subtotal")][
                "bbox_upper_right_x"
            ].max()
            + self.margin
        )

        return description_border_left_x, quantity_border_left_x, amount_border_left_x

    def get_payment_info_bbox(self):
        upper_left_x = self.invoice_df["bbox_upper_left_x"].min() - self.margin
        upper_left_y = (
            self.invoice_df[
                self.invoice_df["text"].str.contains(r"Total[A-z\s]*le", regex=True)
            ]["bbox_lower_left_y"].values[0]
            - self.margin
        )
        lower_right_x = self.invoice_df["bbox_lower_right_x"].max() + self.margin
        lower_right_y = (
            self.invoice_df[self.invoice_df["text"].str.contains("NET AMOUNT PAYABLE")][
                "bbox_lower_right_y"
            ].max()
            + self.margin
        )

        return upper_left_x, upper_left_y, lower_right_x, lower_right_y

    def get_payment_info_df(self):
        (
            upper_left_x,
            upper_left_y,
            lower_right_x,
            lower_right_y,
        ) = self.get_payment_info_bbox()
        payment_info_df = self.invoice_df[
            (self.invoice_df["bbox_upper_left_x"] > upper_left_x)
            & (self.invoice_df["bbox_upper_left_y"] > upper_left_y)
            & (self.invoice_df["bbox_lower_right_x"] < lower_right_x)
            & (self.invoice_df["bbox_lower_right_y"] < lower_right_y)
        ].copy()
        return payment_info_df

    def get_payment_info(self):
        payment_info_df = self.get_payment_info_df()
        payment_info_df["text"] = payment_info_df["text"].apply(
            lambda x: re.sub(r"[^A-Za-z0-9\.\s]", "", x)
        )

        amount_payable_after_tax_right_x = (
            payment_info_df[
                payment_info_df["text"].str.contains("AMOUNT PAYABLE AFTER TAX")
            ]["bbox_upper_right_x"].values[0]
            + self.margin
        )
        payment_info_text_df = payment_info_df[
            payment_info_df["bbox_upper_left_x"] < amount_payable_after_tax_right_x
        ].copy()
        payment_info_payment_df = payment_info_df[
            payment_info_df.index.isin(payment_info_text_df.index) == False
        ].copy()

        payment_info_payment_df["text"] = payment_info_payment_df["text"].apply(
            lambda x: re.sub(r"[^0-9\.]", "", x)
        )
        payment_info_payment_df = payment_info_payment_df[
            payment_info_payment_df["text"] != ""
        ].copy()

        payment_info_payment_df["text"] = payment_info_payment_df["text"].apply(
            lambda x: self.add_decimal_point(x)
        )

        return (payment_info_text_df, payment_info_payment_df)

    def align_payment_info_table_columns(self):
        payment_info_text_df, payment_info_payment_df = self.get_payment_info()

        payment_info_payment_df["text"].to_csv(
            "payment_info_payment_df.csv", index=False
        )
        payment_info_text_df["text"].to_csv("payment_info_text_df.csv", index=False)

        line_items = []
        for i, row in payment_info_text_df.iterrows():
            line_items.append(
                {
                    "payment_info": row["text"],
                    "payment_amount": payment_info_payment_df.iloc[
                        (
                            payment_info_payment_df["bbox_upper_left_y"]
                            - row["bbox_upper_left_y"]
                        )
                        .abs()
                        .argsort()[:1]
                    ]["text"].values[0],
                }
            )
        self.total_payments_info = self.get_total_payments_dict(line_items)

    def check_if_same_line_append_newline_character(self, df):
        df["text_with_newline"] = ""
        for idx in range(df.shape[0] - 1):
            if (
                df.iloc[idx + 1]["bbox_upper_left_y"]
                - df.iloc[idx]["bbox_upper_left_y"]
                > 15
            ):
                df.loc[idx, "text_with_newline"] = df.iloc[idx]["text"] + "\n"
            else:
                df.loc[idx, "text_with_newline"] = df.iloc[idx]["text"]
        # copy the last row
        df.loc[df.shape[0] - 1, "text_with_newline"] = df.iloc[df.shape[0] - 1]["text"]
        return df

    def replace_visit_location(self, text):
        pattern = r"\bVisit\/([\s\S]*)Payment"
        match = re.search(pattern, text)
        if match:
            to_be_replaced = match.group(0).strip().split("Payment")[0].strip()
            replace_with = to_be_replaced.replace("\n ", "")
            return text.replace(to_be_replaced, replace_with)
        else:
            return text

    def check_if_line_starts_with_field_name(self, text):
        for field in self.invoice_key_info_fields:
            if text.startswith(field):
                return field
        return None

    def add_decimal_point(self, text):
        if text.find(".") == -1:
            return text[:-2] + "." + text[-2:]
        else:
            return text

    def clean_invoice_table_columns(self, invoice_table_columns):
        (
            df_invoice_table_codes,
            df_invoice_table_description,
            df_invoice_table_quantity,
            df_invoice_table_amount,
        ) = invoice_table_columns
        df_invoice_table_codes = df_invoice_table_codes[
            df_invoice_table_codes["text"].str.contains("SERVICE CODE") == False
        ].reset_index(drop=True)

        df_invoice_table_description = df_invoice_table_description[
            df_invoice_table_description["text"].str.contains("DESCRIPTION") == False
        ].reset_index(drop=True)

        # remove where text is not a number in quantity and amount
        df_invoice_table_quantity["text"] = df_invoice_table_quantity["text"].apply(
            lambda x: re.sub(r"[^0-9]", "", x)
        )
        df_invoice_table_quantity = df_invoice_table_quantity[
            df_invoice_table_quantity["text"].str.isnumeric()
        ].copy()

        df_invoice_table_amount["text"] = df_invoice_table_amount["text"].apply(
            lambda x: re.sub(r"[^0-9]", "", x)
        )
        df_invoice_table_amount = df_invoice_table_amount[
            df_invoice_table_amount["text"] != ""
        ].copy()

        df_invoice_table_amount["text"] = df_invoice_table_amount["text"].apply(
            lambda x: self.add_decimal_point(x)
        )

        return (
            df_invoice_table_codes.reset_index(drop=True),
            df_invoice_table_description.reset_index(drop=True),
            df_invoice_table_quantity.reset_index(drop=True),
            df_invoice_table_amount.reset_index(drop=True),
        )

    def make_invoice_json(self):
        self.key_info["TaxInvoiceDate"] = self.gst_to_page_info["tax_invoice_date"]
        self.key_info["GSTRegNo"] = self.gst_to_page_info["gst_number"]
        self.key_info["BillType"] = self.gst_to_page_info["bill_type"]
        self.key_info

        invoice_json = {
            "Page_Number": self.gst_to_page_info["page_number"],
            "Table": self.invoice_table,
            "TotalPayments": self.total_payments_info,
            "Key_Values": self.key_info,
        }
        with open("invoice.json", "w") as f:
            json.dump(invoice_json, f, indent=4)
        return invoice_json

    def get_total_payments_dict(self, total_payments_lines):
        total_payments_dict = {}

        for item in total_payments_lines:
            payment_info = re.sub(r"[^A-Za-z]", "", item["payment_info"].title())
            payment_amount = float(item["payment_amount"])
            total_payments_dict[payment_info] = payment_amount

        return total_payments_dict


if __name__ == "__main__":
    pdf_path = "Sample_For_Assignment.pdf"
    pdf_to_ocr = PDFToOCR(pdf_path, page_num=2)
    pdf_to_ocr.perform_ocr()

    full_json_list = []
    for page in pdf_to_ocr.ocr_results[:2]:
        invoice = SingGenHospInvoice()
        invoice.make_invoice_df(page)
        invoice.get_hospital_name()
        invoice.get_gst_to_page_number_info()
        invoice.get_key_info()
        invoice.align_invoice_table_columns()
        invoice.align_payment_info_table_columns()
        invoice_json = invoice.make_invoice_json()

        full_json_list.append(invoice_json)

    with open("json_for_the_pdf.json", "w+") as f:
        json_data = json.dumps(full_json_list, indent=2)
        print(json_data)
        json.dump(json_data, f)
