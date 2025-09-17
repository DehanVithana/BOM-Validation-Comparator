import streamlit as st
import pandas as pd
from io import BytesIO
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

st.title("BOM Consumption Comparison Tool")

uploaded_file = st.file_uploader("Upload your BOM validation Excel file", type=["xlsx"])

if uploaded_file:
    sap_df = pd.read_excel(uploaded_file, sheet_name="SAP", engine="openpyxl")
    plm_df = pd.read_excel(uploaded_file, sheet_name="PLM", engine="openpyxl")

    sap_cols = ['Material', 'Customer Style', 'FG Color', 'Size', 'Cons (qty)']
    plm_cols = ['Material', 'Customer Style', 'Color Name', 'Garment Size', 'Qty(Cons.)']

    sap_subset = sap_df[sap_cols].dropna(subset=['Cons (qty)'])
    plm_subset = plm_df[plm_cols].dropna(subset=['Qty(Cons.)'])

    def compute_similarity(row1, row2):
        score = 0
        score += fuzz.token_sort_ratio(str(row1['Material']), str(row2['Material']))
        score += fuzz.token_sort_ratio(str(row1['Customer Style']), str(row2['Customer Style']))
        score += fuzz.token_sort_ratio(str(row1['FG Color']), str(row2['Color Name']))
        score += fuzz.token_sort_ratio(str(row1['Size']), str(row2['Garment Size']))
        return score / 4

    matches = []
    mismatches = []

    for plm_index, plm_row in plm_subset.iterrows():
        best_match = None
        best_score = 0
        for sap_index, sap_row in sap_subset.iterrows():
            score = compute_similarity(sap_row, plm_row)
            if score > best_score:
                best_score = score
                best_match = sap_row
        if best_score >= 80:
            consumption_match = abs(best_match['Cons (qty)'] - plm_row['Qty(Cons.)']) < 0.01
            match_record = {
                'PLM Index': plm_index,
                'SAP Material': best_match['Material'],
                'SAP Style': best_match['Customer Style'],
                'SAP Color': best_match['FG Color'],
                'SAP Size': best_match['Size'],
                'SAP Cons (qty)': best_match['Cons (qty)'],
                'PLM Material': plm_row['Material'],
                'PLM Style': plm_row['Customer Style'],
                'PLM Color': plm_row['Color Name'],
                'PLM Size': plm_row['Garment Size'],
                'PLM Qty(Cons.)': plm_row['Qty(Cons.)'],
                'Similarity Score': best_score,
                'Consumption Match': consumption_match
            }
            matches.append(match_record)
            if not consumption_match:
                mismatches.append(match_record)

    if matches:
        report_df = pd.DataFrame(matches)
        st.write("### Comparison Results")
        st.dataframe(report_df)

        if mismatches:
            mismatch_df = pd.DataFrame(mismatches)
            st.write("### Mismatches in Consumption")
            fig, ax = plt.subplots()
            ax.bar(mismatch_df.index, mismatch_df['SAP Cons (qty)'], label='SAP Cons (qty)', alpha=0.7)
            ax.bar(mismatch_df.index, mismatch_df['PLM Qty(Cons.)'], label='PLM Qty(Cons.)', alpha=0.7)
            ax.set_xlabel("Mismatch Index")
            ax.set_ylabel("Consumption Quantity")
            ax.set_title("Consumption Mismatches")
            ax.legend()
            st.pyplot(fig)

        output = BytesIO()
        report_df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Comparison Report",
            data=output,
            file_name="consumption_comparison_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No matches found above the similarity threshold.")

st.markdown("""
**Instructions:**  
- Upload your BOM validation Excel file with SAP, PLM, and Size Chart sheets.
- The app will compare the Consumption columns using fuzzy matching, visualize mismatches, and let you download the results.
""")
