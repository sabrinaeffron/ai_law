import re
import pandas as pd
import random

CURRENT_YEAR = 2025
GRAD_YEARS = [2022, 2007, 1992, 1977] # approx age 25, 40, 55, 70
AGE_RANGE = 5

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
random.seed(3000)

def extract_years(text):
    years = [int(m.group()) for m in YEAR_RE.finditer(text)]
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
            if max_year > 2025:
                continue

            for i in range(len(GRAD_YEARS)):
                year = GRAD_YEARS[i] + random.randint(0, AGE_RANGE) # grad year in range
                shift = year - earliest # number to add to all dates
                if shift > 0 and (max_year + shift > CURRENT_YEAR):
                    shift = CURRENT_YEAR - max_year # maximum shift
                    year = earliest + shift
                text = shift_years(row["Resume_str"], shift)
                records.append({
                    "base_id": row["ID"],
                    "variant": i,
                    "shift_years": shift,
                    "grad_year": year,
                    "resume_text": text,
                    "category": row["Category"]
                })
    out = pd.DataFrame(records)
    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

if __name__ == "__main__":
    in_csv  = "Resume.csv"
    out_csv = "Resumes_cleaned.csv"
    main(in_csv, out_csv)
