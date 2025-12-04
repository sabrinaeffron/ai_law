import re
import pandas as pd
import random

CURRENT_YEAR = 2025
YOUNG_TARGET_GRAD = 2012 # approx age 35
OLD_TARGET_GRAD = 1997 # approx age 50
HALF_RANGE = 5

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
random.seed(3000)

def extract_years(text):
    years = [int(m.group()) for m in YEAR_RE.finditer(text)]
    years = [y for y in years if 1980 <= y <= CURRENT_YEAR]
    return sorted(set(years))

def shift_years(text, shift):
    def repl(m):
        return str(int(m.group()) + shift)
    return YEAR_RE.sub(repl, text)

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    records = []

    for _, row in df.iterrows():
        # Only take from these three categories
        if row['Category'] == 'INFORMATION-TECHNOLOGY' or row['Category'] == 'BUSINESS-DEVELOPMENT' or row['Category'] == 'ACCOUNTANT':
            # Extract all years from the resume
            years = extract_years(row["Resume_str"])
            if not years:
                continue
            earliest = min(years)
            max_year = max(years)

            # Younger version
            if earliest > YOUNG_TARGET_GRAD + HALF_RANGE or earliest < YOUNG_TARGET_GRAD - HALF_RANGE:
                younger_year = YOUNG_TARGET_GRAD - random.randint(-HALF_RANGE, HALF_RANGE)
                younger_shift = younger_year - earliest
                # Clamp younger shift to avoid future years
                if younger_shift > 0 and (max_year + younger_shift > CURRENT_YEAR):
                    younger_shift = CURRENT_YEAR - max_year
                    if younger_shift < 0:
                        younger_shift = 0
                younger_text = shift_years(row["Resume_str"], younger_shift)
            else:
                younger_shift = 0
                younger_text = row["Resume_str"]
            records.append({
                "base_id": row["ID"],
                "variant": "younger",
                "shift_years": younger_shift,
                "resume_text": younger_text,
                "category": row["Category"]
            })

            # Older version
            if earliest > OLD_TARGET_GRAD + HALF_RANGE or earliest < OLD_TARGET_GRAD - HALF_RANGE:
                older_year = OLD_TARGET_GRAD - random.randint(-HALF_RANGE, HALF_RANGE)
                older_shift = older_year - earliest
                older_text = shift_years(row["Resume_str"], older_shift)
            else:
                older_shift = 0
                older_text = row["Resume_str"]
            records.append({
                "base_id": row["ID"],
                "variant": "older",
                "shift_years": older_shift,
                "resume_text": older_text,
                "category": row["Category"]
            })

    out = pd.DataFrame(records)
    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

if __name__ == "__main__":
    in_csv  = "Resume.csv"
    out_csv = "Resumes_cleaned.csv"
    main(in_csv, out_csv)
