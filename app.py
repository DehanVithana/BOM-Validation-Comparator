'''
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
'''
import streamlit as st
import pandas as pd
import plotly.express as px
from thefuzz import process, fuzz
from io import BytesIO

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="SAP & PLM Consumption Analysis")

st.markdown("""
<style>
    .reportview-container {
        background: #f5f5f5;
    }
    .main > div {
        padding-top: 2rem;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 2rem;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #007bff;
        color: #007bff;
    }
    .stButton>button:hover {
        border-color: #0056b3;
        color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def to_excel(df):
    """Converts a DataFrame to an Excel file in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def run_matching_logic(sap_df, plm_df, size_chart_df, threshold=70):
    """
    Core function to process, match, and compare the data from SAP and PLM.
    """
    # 1. Pre-process DataFrames
    # Clean up column names
    sap_df.columns = sap_df.columns.str.strip()
    plm_df.columns = plm_df.columns.str.strip()
    size_chart_df.columns = size_chart_df.columns.str.strip()

    # Rename key columns for consistency
    sap_df.rename(columns={
        'Material': 'Material_ID',
        'Customer Style': 'Style',
        'Vendor Reference': 'Vendor_Ref',
        'Comp.Qty.': 'SAP_Consumption',
        'Component Grv': 'Size'
    }, inplace=True)

    plm_df.rename(columns={
        'Material': 'Material_ID',
        'Customer Style': 'Style',
        'Vendor Reference': 'Vendor_Ref',
        'Qty(Cons.)': 'PLM_Consumption',
        'Size Split': 'Size_Split_ID'
    }, inplace=True)

    size_chart_df.rename(columns={'Size Split': 'Size_Split_ID', 'Size List': 'Size'}, inplace=True)
    
    # Bridge PLM and Size Chart
    # This connects the PLM's size ID to actual size names (e.g., S, M, L)
    plm_merged_df = pd.merge(plm_df, size_chart_df[['Size_Split_ID', 'Size']], on='Size_Split_ID', how='left')

    # Explode sizes: A single PLM row can apply to multiple sizes (e.g., "S,M").
    # We create a separate row for each size.
    plm_merged_df['Size'] = plm_merged_df['Size'].str.split(',')
    plm_exploded_df = plm_merged_df.explode('Size')
    plm_exploded_df['Size'] = plm_exploded_df['Size'].str.strip()

    # Create a composite key for fuzzy matching. This string combines several
    # relevant attributes to give a better basis for comparison.
    plm_exploded_df['Match_Key'] = plm_exploded_df['Vendor_Ref'].astype(str) + "_" + \
                                   plm_exploded_df['Color Name'].astype(str) + "_" + \
                                   plm_exploded_df['Position'].astype(str)

    sap_df['Match_Key'] = sap_df['Vendor_Ref'].astype(str) + "_" + \
                          sap_df['Component Col. Des.'].astype(str) + "_" + \
                          sap_df['Head Material Group'].astype(str)
                          
    # 2. Perform Fuzzy Matching
    results = []
    matched_plm_indices = set()

    # Group data by Style and Material ID to reduce search space
    grouped_plm = plm_exploded_df.groupby(['Style', 'Material_ID'])

    for index, sap_row in sap_df.iterrows():
        # Find corresponding group in PLM data
        try:
            plm_group = grouped_plm.get_group((sap_row['Style'], sap_row['Material_ID']))
            
            # Filter PLM group by the same size as the SAP row
            plm_group_filtered = plm_group[plm_group['Size'] == sap_row['Size']]

            if not plm_group_filtered.empty:
                # Use thefuzz to find the best match within the filtered group
                best_match = process.extractOne(
                    sap_row['Match_Key'],
                    plm_group_filtered['Match_Key'],
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold
                )

                if best_match:
                    match_key, score, plm_index = best_match
                    plm_row = plm_group_filtered.loc[plm_index]
                    
                    # Combine SAP and best PLM match
                    combined_row = {**sap_row.to_dict(), **plm_row.add_prefix('PLM_').to_dict()}
                    combined_row['Similarity_Score'] = score
                    combined_row['Status'] = 'Matched'
                    results.append(combined_row)
                    matched_plm_indices.add(plm_index)
                    continue # Move to the next SAP row
        
        except KeyError:
            # No matching group found in PLM data for the SAP row's style/material
            pass
        
        # If no match was found, add SAP row as unmatched
        unmatched_sap = sap_row.to_dict()
        unmatched_sap['Status'] = 'Unmatched SAP'
        results.append(unmatched_sap)

    # Add unmatched PLM rows
    unmatched_plm_df = plm_exploded_df[~plm_exploded_df.index.isin(matched_plm_indices)]
    for index, plm_row in unmatched_plm_df.iterrows():
        unmatched_plm = plm_row.add_prefix('PLM_').to_dict()
        unmatched_plm['Status'] = 'Unmatched PLM'
        results.append(unmatched_plm)

    return pd.DataFrame(results)


# --- Streamlit UI ---

st.title("📊 SAP vs. PLM Consumption Comparison")
st.write("""
    Upload your SAP, PLM, and Size Chart data from Excel to compare consumption quantities. 
    The application uses a similarity score to match records that may not have identical descriptions.
""")

# --- Sidebar for Uploads and Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    st.subheader("1. Upload Data Files")
    sap_file = st.file_uploader("Upload SAP Data", type=['csv', 'xlsx'])
    plm_file = st.file_uploader("Upload PLM Data", type=['csv', 'xlsx'])
    size_chart_file = st.file_uploader("Upload Size Chart Data", type=['csv', 'xlsx'])

    st.subheader("2. Set Matching Threshold")
    similarity_threshold = st.slider(
        "Similarity Score Threshold (%)",
        min_value=50,
        max_value=100,
        value=70,
        help="A higher threshold requires a closer match between item descriptions."
    )

if sap_file and plm_file and size_chart_file:
    try:
        sap_df = pd.read_csv(sap_file)
        plm_df = pd.read_csv(plm_file)
        size_chart_df = pd.read_csv(size_chart_file)

        # Process data and run matching logic
        final_report_df = run_matching_logic(sap_df, plm_df, size_chart_df, similarity_threshold)

        # Filter for different views
        matched_df = final_report_df[final_report_df['Status'] == 'Matched'].copy()
        unmatched_sap_df = final_report_df[final_report_df['Status'] == 'Unmatched SAP']
        unmatched_plm_df = final_report_df[final_report_df['Status'] == 'Unmatched PLM']

        # --- Main Page Display ---
        st.header("📈 Matching Summary")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Matched Records", f"{len(matched_df)}")
        col2.metric("❌ Unmatched SAP Records", f"{len(unmatched_sap_df)}")
        col3.metric("❌ Unmatched PLM Records", f"{len(unmatched_plm_df)}")
        
        if not matched_df.empty:
            matched_df['Consumption_Diff'] = matched_df['SAP_Consumption'] - matched_df['PLM_Consumption']
            
            st.header("📊 Visualizations for Matched Records")
            
            # Consumption Comparison Bar Chart
            total_sap_cons = matched_df['SAP_Consumption'].sum()
            total_plm_cons = matched_df['PLM_Consumption'].sum()
            
            fig_bar = px.bar(
                x=['SAP Total Consumption', 'PLM Total Consumption'],
                y=[total_sap_cons, total_plm_cons],
                labels={'x': 'System', 'y': 'Total Consumption'},
                title='Total Consumption Comparison (SAP vs. PLM)',
                color=['SAP Total Consumption', 'PLM Total Consumption'],
                color_discrete_map={
                    'SAP Total Consumption': 'rgba(255, 75, 75, 0.8)',
                    'PLM Total Consumption': 'rgba(75, 75, 255, 0.8)'
                },
                text_auto='.2f'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Scatter Plot
            fig_scatter = px.scatter(
                matched_df,
                x='SAP_Consumption',
                y='PLM_Consumption',
                color='Similarity_Score',
                hover_data=['Style', 'Material_ID', 'Vendor_Ref', 'Size'],
                title='SAP vs. PLM Consumption per Matched Item'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.header("📄 Detailed Report")
        
        # Download button for the full report
        excel_data = to_excel(final_report_df)
        st.download_button(
            label="📥 Download Full Report (Excel)",
            data=excel_data,
            file_name=f"sap_plm_comparison_report_{similarity_threshold}pct.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Tabs for detailed data views
        tab1, tab2, tab3 = st.tabs(["✅ Matched Records", "❌ Unmatched SAP", "❌ Unmatched PLM"])

        with tab1:
            st.dataframe(matched_df)
        with tab2:
            st.dataframe(unmatched_sap_df)
        with tab3:
            st.dataframe(unmatched_plm_df)
            
    except Exception as e:
        st.error(f"An error occurred while processing the files: {e}")
        st.warning("Please ensure the uploaded files are in the correct format and have the expected columns.")

else:
    st.info("👋 Welcome! Please upload all three required files in the sidebar to begin the analysis.")
