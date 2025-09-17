import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import io

def load_data(uploaded_file):
    """Loads a CSV file into a pandas DataFrame."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

def find_best_match(plm_row, sap_df, threshold=80):
    """
    Finds the best matching SAP row for a given PLM row using fuzzy matching.
    Returns the matched row, similarity score, and combined key.
    """
    plm_key = f"{plm_row['Customer Style']}|{plm_row['Color Reference']}|{plm_row['Vendor Reference']}"
    
    best_match_row = None
    best_score = 0
    
    for _, sap_row in sap_df.iterrows():
        # Corrected line to use the renamed column 'Color Reference'
        sap_key = f"{sap_row['Customer Style']}|{sap_row['Color Reference']}|{sap_row['Vendor Reference']}"
        
        # Use token set ratio for robust matching
        score = fuzz.token_set_ratio(plm_key.lower(), sap_key.lower())
        
        if score > best_score:
            best_score = score
            best_match_row = sap_row
    
    if best_score >= threshold:
        return best_match_row, best_score, plm_key
    else:
        return None, best_score, plm_key

def main():
    """Main Streamlit app function."""
    st.title('BOM Validation App')
    st.markdown('Upload the SAP, PLM, and Size Chart CSV files to compare consumption data.')

    # File Uploaders
    sap_file = st.file_uploader("Upload SAP (SAP.csv)", type=['csv'])
    plm_file = st.file_uploader("Upload PLM (PLM.csv)", type=['csv'])
    size_chart_file = st.file_uploader("Upload Size Chart (Size chart.csv)", type=['csv'])

    if sap_file and plm_file:
        sap_df = load_data(sap_file)
        plm_df = load_data(plm_file)
        size_df = load_data(size_chart_file)

        if sap_df is not None and plm_df is not None:
            st.success("Files loaded successfully!")
            
            # Standardize columns
            sap_df.rename(columns={'Consumption': 'Cons (qty)', 'Comp. Colour': 'Color Reference', 'Material': 'Component'}, inplace=True)
            plm_df.rename(columns={'Qty(Cons.)': 'Consumption', 'Material': 'Component'}, inplace=True)

            # --- Matching and Comparison ---
            results = []
            for _, plm_row in plm_df.iterrows():
                best_match_sap, score, plm_key = find_best_match(plm_row, sap_df)
                
                plm_consumption = pd.to_numeric(plm_row.get('Consumption', 0), errors='coerce')
                sap_consumption = pd.to_numeric(best_match_sap.get('Cons (qty)', 0) if best_match_sap is not None else 0, errors='coerce')
                
                comparison = ""
                difference = 0
                if best_match_sap is not None:
                    if not pd.isna(plm_consumption) and not pd.isna(sap_consumption):
                        difference = plm_consumption - sap_consumption
                        if abs(difference) > 0.001:  # Allow for small floating point errors
                            comparison = "MISMATCH"
                        else:
                            comparison = "MATCH"
                    else:
                        comparison = "MISSING VALUE"
                else:
                    comparison = "NO MATCH"
                
                results.append({
                    'PLM Style': plm_row.get('Customer Style'),
                    'PLM Color': plm_row.get('Color Reference'),
                    'PLM Component': plm_row.get('Vendor Reference'),
                    'PLM Consumption': plm_consumption,
                    'SAP Style': best_match_sap.get('Customer Style') if best_match_sap is not None else 'N/A',
                    'SAP Color': best_match_sap.get('Color Reference') if best_match_sap is not None else 'N/A',
                    'SAP Component': best_match_sap.get('Vendor Reference') if best_match_sap is not None else 'N/A',
                    'SAP Consumption': sap_consumption,
                    'Similarity Score': score,
                    'Consumption Comparison': comparison,
                    'Consumption Difference': difference
                })
            
            results_df = pd.DataFrame(results)
            
            st.header("Comparison Results")
            st.dataframe(results_df)

            # --- Summary and Visualization ---
            st.header("Summary")
            
            mismatched_count = len(results_df[results_df['Consumption Comparison'] == 'MISMATCH'])
            matched_count = len(results_df[results_df['Consumption Comparison'] == 'MATCH'])
            no_match_count = len(results_df[results_df['Consumption Comparison'] == 'NO MATCH'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", matched_count)
            with col2:
                st.metric("Total Mismatches", mismatched_count)
            with col3:
                st.metric("Rows with No Match", no_match_count)

            st.subheader("Mismatched Rows")
            mismatched_df = results_df[results_df['Consumption Comparison'].isin(['MISMATCH', 'NO MATCH', 'MISSING VALUE'])]
            st.dataframe(mismatched_df)

            # --- Download Button ---
            csv_output = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Comparison Report",
                data=csv_output,
                file_name="bom_comparison_report.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    main()
