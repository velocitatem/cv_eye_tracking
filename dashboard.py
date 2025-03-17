# dashboard.py

""" Streamlit dashboard for visualizing attention tracking results. """

# Importing Libs

import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go


# Config

st.set_page_config(
    page_title="Face Attention Tracker",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the external CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# custom CSS file
local_css("style.css")

# LOAD THE DATA

@st.cache_data
def load_data():
    if os.path.exists("attention_analysis_results.pkl"):
        with open("attention_analysis_results.pkl", "rb") as f:
            return pickle.load(f)
    else:
        st.error("Could not find attention analysis results. Please run the analysis script first.")
        return None
    
# GALLERY VIEW OF THE PERSONS

def display_person_gallery(results, selected_person=None, key_prefix="overview"):
    """Display a gallery of all people with their attention scores as clickable cards."""
    person_avg_attention = results["person_avg_attention"]
    person_images = results["person_images"]

    # Sort people by average attention
    sorted_people = sorted(person_avg_attention.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(min(5, len(sorted_people)))

    for i, (person_id, avg_attention) in enumerate(sorted_people):
        with cols[i % len(cols)]:
            if st.button(f"Person {person_id}", key=f"{key_prefix}_person_{person_id}_{i}"):
                st.session_state.selected_person = person_id

            if selected_person is not None and int(selected_person) == person_id:
                st.markdown(f"""
                <div style="border:3px solid #FF5733; padding:5px; border-radius:5px;">
                    <h4 style="text-align:center;">Person {person_id}</h4>
                    <p style="text-align:center; font-weight:bold;">Attention: {avg_attention:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="person-card">
                    <h4 style="text-align:center;">Person {person_id}</h4>
                    <p style="text-align:center;">Attention: {avg_attention:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            if person_images.get(person_id) and len(person_images[person_id]) > 0:
                img = person_images[person_id][len(person_images[person_id]) // 2]
                st.image(img)


def plot_attention_time_series(results, selected_person=None):
    """
    If 'selected_person' is None, show lines for ALL people.
    """
    person_attention_series = results["person_attention_series"]
    person_timestamps = results["person_timestamps"]

    if selected_person is not None:
        pid = int(selected_person)
        df = pd.DataFrame({
            "Frame": person_timestamps[pid],
            "Attention": person_attention_series[pid],
            "Person": [pid]*len(person_timestamps[pid])
        })
        title = f"Attention Over Time for Person {pid}"
    else:
        data_list = []
        for pid, timestamps in person_timestamps.items():
            for t, att in zip(timestamps, person_attention_series[pid]):
                data_list.append({"Frame": t, "Attention": att, "Person": pid})
        df = pd.DataFrame(data_list)
        title = "Attention Over Time (All People)"

    fig = px.line(
        df,
        x="Frame",
        y="Attention",
        color="Person",
        markers=True,
        title=title,
        labels={"Frame": "Frame Number", "Attention": "Attention Score", "Person": "Person ID"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    if selected_person is not None and len(df) > 0:
        avg_val = df["Attention"].mean()
        fig.add_hline(
            y=avg_val,
            line_dash="dash",
            annotation_text=f"Avg: {avg_val:.2f}",
            annotation_position="top left"
        )

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig

def plot_attention_distribution(results, selected_person=None):
    """
    If 'selected_person' is None, We show a boxplot for all persons;
    otherwise, we show a histogram for that person.
    """
    person_attention_series = results["person_attention_series"]

    if selected_person is not None:
        pid = int(selected_person)
        data = person_attention_series[pid]
        title = f"Attention Distribution for Person {pid}"
        df = pd.DataFrame({"Attention": data, "Person": [pid]*len(data)})
        fig = px.histogram(
            df,
            x="Attention",
            nbins=30,
            marginal="rug",
            title=title,
            labels={"Attention": "Attention Score"},
            color_discrete_sequence=["#FF5733"]
        )
    else:
        data_list = []
        for pid, values in person_attention_series.items():
            for val in values:
                data_list.append({"Person": pid, "Attention": val})
        df = pd.DataFrame(data_list)
        title = "Overall Attention Distribution (All People)"
        fig = px.box(
            df,
            x="Person",
            y="Attention",
            points="all",
            title=title,
            labels={"Person": "Person ID", "Attention": "Attention Score"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    return fig

def create_summary_stats(results, selected_person=None):
    """Create a summary statistics DataFrame."""
    person_attention_series = results["person_attention_series"]
    stats_data = []
    if selected_person is not None:
        person_id = int(selected_person)
        values = person_attention_series[person_id]
        stats_data.append({
            "Person ID": person_id,
            "Average Attention": np.mean(values),
            "Min Attention": np.min(values),
            "Max Attention": np.max(values),
            "Attention Range": np.max(values) - np.min(values),
            "Attention Variance": np.var(values),
            "Number of Appearances": len(values)
        })
    else:
        for pid, values in person_attention_series.items():
            stats_data.append({
                "Person ID": pid,
                "Average Attention": np.mean(values),
                "Min Attention": np.min(values),
                "Max Attention": np.max(values),
                "Attention Range": np.max(values) - np.min(values),
                "Attention Variance": np.var(values),
                "Number of Appearances": len(values)
            })
    stats_df = pd.DataFrame(stats_data)
    return stats_df.sort_values("Average Attention", ascending=False)

def main():
    st.title("👁️ Face Attention Tracking Dashboard")
    results = load_data()
    if results is None:
        return

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    person_ids = list(results["person_avg_attention"].keys())
    person_ids_str = [str(pid) for pid in person_ids]
    selected_person_sidebar = st.sidebar.selectbox("Select Person for Analysis", options=["All"] + person_ids_str)
    if selected_person_sidebar != "All":
        st.session_state.selected_person = int(selected_person_sidebar)
    else:
        st.session_state.selected_person = None

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This dashboard visualizes attention tracking data for individuals across multiple video frames. "
        "Click on a person card or use the sidebar to view detailed analysis."
    )

    # Main layout with Tabs
    tab1, tab2, tab3 = st.tabs(["Overall Summary", "Individual Analysis", "Advanced Visualizations"])

    # Overall Summary Tab
    with tab1:
        st.header("📊 Overall Attention Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total People", len(results["person_avg_attention"]))
        with col2:
            total_observations = sum(len(v) for v in results["person_attention_series"].values())
            st.metric("Total Observations", total_observations)
        with col3:
            avg_attention = np.mean([v for values in results["person_attention_series"].values() for v in values])
            st.metric("Average Attention", f"{avg_attention:.2f}")

        st.subheader("👥 People Overview")
        display_person_gallery(results, st.session_state.get("selected_person"), key_prefix="overview")

        st.subheader("📈 Attention Over Time (All People)")
        fig_time_all = plot_attention_time_series(results)
        st.plotly_chart(fig_time_all, use_container_width=True)

        st.subheader("📋 Detailed Statistics")
        stats_df = create_summary_stats(results)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("📊 Overall Attention Distribution")
        fig_dist_all = plot_attention_distribution(results)
        st.plotly_chart(fig_dist_all, use_container_width=True)

        csv = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary Stats as CSV", data=csv, file_name='summary_stats.csv', mime='text/csv')

    # Individual Analysis Tab
    with tab2:
        if st.session_state.get("selected_person") is None:
            st.header("👥 All People Analysis")
            display_person_gallery(results, key_prefix="individual_all")
            st.subheader("📈 Attention Over Time")
            fig_time_all_2 = plot_attention_time_series(results)
            st.plotly_chart(fig_time_all_2, use_container_width=True)
            st.subheader("📋 Comparative Statistics")
            stats_df = create_summary_stats(results)
            st.dataframe(stats_df, use_container_width=True)
        else:
            person_id = st.session_state.selected_person
            st.header(f"👤 Person {person_id} Analysis")
            display_person_gallery(results, selected_person=person_id, key_prefix="individual")
            col1, col2, col3 = st.columns(3)
            values = results["person_attention_series"][person_id]
            avg = np.mean(values)
            with col1:
                st.metric("Average Attention", f"{avg:.2f}")
            with col2:
                st.metric("Appearances", len(values))
            with col3:
                variance = np.var(values)
                st.metric("Attention Variance", f"{variance:.4f}")
            st.subheader("📈 Attention Over Time")
            fig_time_person = plot_attention_time_series(results, selected_person=person_id)
            st.plotly_chart(fig_time_person, use_container_width=True)
            st.subheader("📊 Attention Distribution")
            fig_dist_person = plot_attention_distribution(results, selected_person=person_id)
            st.plotly_chart(fig_dist_person, use_container_width=True)
            st.subheader("📋 Detailed Statistics")
            stats_df = create_summary_stats(results, selected_person=person_id)
            st.dataframe(stats_df, use_container_width=True)
            st.subheader("🖼️ All Appearances")
            person_images = results["person_images"][person_id]
            img_index = st.slider("Select Image", 0, len(person_images)-1, 0, key=f"slider_{person_id}")
            st.image(person_images[img_index], caption=f"Frame {results['person_timestamps'][person_id][img_index]}")

    # Advanced Visualizations Tab
    with tab3:
        st.header("🚀 Advanced Visualizations")
        attention_values_all = [v for values in results["person_attention_series"].values() for v in values]
        df_heat = pd.DataFrame({"Attention": attention_values_all})
        fig_heat = px.density_heatmap(
            df_heat,
            x="Attention",
            nbinsx=30,
            title="Attention Density Heatmap",
            labels={"Attention": "Attention Score"},
            color_continuous_scale="Plasma"
        )
        fig_heat.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()
