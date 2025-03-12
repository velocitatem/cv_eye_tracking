import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="Face Attention Tracker",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stPlotlyChart {
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.person-card {
    border: 1px solid #e6e6e6;
    padding: 10px;
    border-radius: 5px;
    background-color: #f9f9f9;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the processed attention data."""
    if os.path.exists("attention_analysis_results.pkl"):
        with open("attention_analysis_results.pkl", "rb") as f:
            results = pickle.load(f)
        return results
    else:
        st.error("Could not find attention analysis results. Please run the analysis script first.")
        return None

def display_person_gallery(results, selected_person=None):
    """Display a gallery of all people with their attention scores."""
    person_avg_attention = results["person_avg_attention"]
    person_images = results["person_images"]

    # Sort people by average attention
    sorted_people = sorted(person_avg_attention.items(), key=lambda x: x[1], reverse=True)

    cols = st.columns(min(5, len(sorted_people)))

    for i, (person_id, avg_attention) in enumerate(sorted_people):
        with cols[i % len(cols)]:
            # Get a representative image
            if len(person_images[person_id]) > 0:
                img = person_images[person_id][len(person_images[person_id])//2]

                # Create a border effect for selected person
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

                # Display the image
                st.image(img, use_container_width=False)

def plot_attention_time_series(results, selected_person=None):
    """Plot attention over time for selected or all people."""
    person_attention_series = results["person_attention_series"]
    person_timestamps = results["person_timestamps"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all people or just the selected one
    if selected_person is not None:
        person_id = int(selected_person)
        timestamps = person_timestamps[person_id]
        attention_values = person_attention_series[person_id]
        ax.plot(timestamps, attention_values, 'o-', linewidth=3,
                label=f"Person {person_id}", color='#FF5733')

        # Add average line
        avg = np.mean(attention_values)
        ax.axhline(y=avg, color='#33FF57', linestyle='--',
                  label=f"Average: {avg:.2f}")
    else:
        # Plot all people with different colors
        for person_id, timestamps in person_timestamps.items():
            attention_values = person_attention_series[person_id]
            ax.plot(timestamps, attention_values, 'o-', alpha=0.7,
                    label=f"Person {person_id}")

    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Attention Score", fontsize=12)
    ax.set_title("Attention Scores Through Time", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig

def plot_attention_distribution(results, selected_person=None):
    """Plot the distribution of attention scores."""
    person_attention_series = results["person_attention_series"]

    # Prepare data for plotting
    if selected_person is not None:
        person_id = int(selected_person)
        data = person_attention_series[person_id]
        title = f"Attention Distribution for Person {person_id}"
    else:
        # Combine all attention scores
        data = []
        for values in person_attention_series.values():
            data.extend(values)
        title = "Overall Attention Distribution"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, kde=True, ax=ax)
    ax.set_xlabel("Attention Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

def create_summary_stats(results, selected_person=None):
    """Create a summary statistics dataframe."""
    person_attention_series = results["person_attention_series"]
    person_avg_attention = results["person_avg_attention"]

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
        for person_id, values in person_attention_series.items():
            stats_data.append({
                "Person ID": person_id,
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

    # Load the processed data
    results = load_data()
    if results is None:
        return

    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")

    # Person selection
    person_ids = list(results["person_avg_attention"].keys())
    person_ids = [str(id) for id in person_ids]

    view_option = st.sidebar.radio(
        "Select View",
        ["Overall Summary", "Individual Person Analysis"]
    )

    selected_person = None
    if view_option == "Individual Person Analysis":
        selected_person = st.sidebar.selectbox(
            "Select Person",
            ["All"] + person_ids
        )
        if selected_person == "All":
            selected_person = None

    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This dashboard visualizes attention tracking data for individuals "
        "across multiple video frames. Select a specific person to see their "
        "detailed attention patterns."
    )

    # Main content
    if view_option == "Overall Summary":
        # Overview section
        st.header("📊 Overall Attention Summary")

        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total People", len(results["person_avg_attention"]))

        with col2:
            total_observations = sum(len(v) for v in results["person_attention_series"].values())
            st.metric("Total Observations", total_observations)

        with col3:
            avg_attention = np.mean([v for values in results["person_attention_series"].values() for v in values])
            st.metric("Average Attention", f"{avg_attention:.2f}")

        # People gallery
        st.subheader("👥 People Overview")
        display_person_gallery(results)

        # Attention time series
        st.subheader("📈 Attention Over Time (All People)")
        st.pyplot(plot_attention_time_series(results))

        # Summary statistics
        st.subheader("📋 Detailed Statistics")
        stats_df = create_summary_stats(results)
        st.dataframe(stats_df, use_container_width=True)

        # Attention distribution
        st.subheader("📊 Overall Attention Distribution")
        st.pyplot(plot_attention_distribution(results))

    else:  # Individual person analysis
        if selected_person is None:
            st.header("👥 All People Analysis")

            # Display gallery with all people
            display_person_gallery(results)

            # Time series plot for all
            st.subheader("📈 Attention Time Series")
            st.pyplot(plot_attention_time_series(results))

            # Summary statistics
            st.subheader("📋 Comparative Statistics")
            stats_df = create_summary_stats(results)
            st.dataframe(stats_df, use_container_width=True)

        else:
            person_id = int(selected_person)
            st.header(f"👤 Person {person_id} Analysis")

            # Display gallery highlighting selected person
            display_person_gallery(results, selected_person)

            # Person metrics
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

            # Time series plot for selected person
            st.subheader("📈 Attention Over Time")
            st.pyplot(plot_attention_time_series(results, selected_person))

            # Attention distribution
            st.subheader("📊 Attention Distribution")
            st.pyplot(plot_attention_distribution(results, selected_person))

            # Detailed stats
            st.subheader("📋 Detailed Statistics")
            stats_df = create_summary_stats(results, selected_person)
            st.dataframe(stats_df, use_container_width=True)

            # Show all images of this person in a grid
            st.subheader("🖼️ All Appearances")
            person_images = results["person_images"][person_id]

            # Display images in a grid
            image_cols = st.columns(5)
            for i, img in enumerate(person_images):
                with image_cols[i % 5]:
                    st.image(img, caption=f"Frame {results['person_timestamps'][person_id][i]}")

if __name__ == "__main__":
    main()
