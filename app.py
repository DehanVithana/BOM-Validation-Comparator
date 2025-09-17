import streamlit as st
import pandas as pd
from io import BytesIO
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

st.title("BOM Consumption Comparison Tool")

uploaded_file = st.file_uploader("Upload your BOM validation Excel file", type=["xlsx"])

if uploaded_file:
    sap_df = pd.read_excel(uploaded_file, sheet_name="SAP")
    plm_df = pd.read_excel(uploaded_file, sheet_name="PLM")

    # Show actual column names for debugging
    st.write("SAP columns:", list(sap_df.columns))
    st.write("PLM columns:", list(plm_df.columns))

    # Standardize column names (strip spaces, lower case)
    sap_df.columns = sap_df.columns.str.strip().str.lower()
    plm_df.columns = plm_df.columns.str.strip().str.lower()

    # Define expected columns (lowercase, stripped)
    sap_cols = ['material', 'customer style', 'fg color', 'size', 'cons (qty)']
    plm_cols = ['material', 'customer style', 'color name', 'garment size', 'qty(cons.)']

    # Check for missing columns
    missing_sap = [col for col in sap_cols if col not in sap_df.columns]
    missing_plm = [col for col in plm_cols if col not in plm_df.columns]
    if missing_sap or missing_plm:
        st.error(f"Missing columns. SAP: {missing_sap}, PLM: {missing_plm}")
    else:
        sap_subset = sap_df[sap_cols].dropna(subset=['cons (qty)'])
        plm_subset = plm_df[plm_cols].dropna(subset=['qty(cons.)'])

        def compute_similarity(row1, row2):
            score = 0
            score += fuzz.token_sort_ratio(str(row1['material']), str(row2['material']))
            score += fuzz.token_sort_ratio(str(row1['customer style']), str(row2['customer style']))
            score += fuzz.token_sort_ratio(str(row1['fg color']), str(row2['color name']))
            score += fuzz.token_sort_ratio(str(row1['size']), str(row2['garment size']))
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
                consumption_match = abs(best_match['cons (qty)'] - plm_row['qty(cons.)']) < 0.01
                match_record = {
                    'PLM Index': plm_index,
                    'SAP Material': best_match['material'],
                    'SAP Style': best_match['customer style'],
                    'SAP Color': best_match['fg color'],
                    'SAP Size': best_match['size'],
                    'SAP Cons (qty)': best_match['cons (qty)'],
                    'PLM Material': plm_row['material'],
                    'PLM Style': plm_row['customer style'],
                    'PLM Color': plm_row['color name'],
                    'PLM Size': plm_row['garment size'],
                    'PLM Qty(Cons.)': plm_row['qty(cons.)'],
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
