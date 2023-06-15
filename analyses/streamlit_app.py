import streamlit as st
import pandas as pd


def main():
    st.set_page_config(layout="wide")
    st.title("Parent Child Consistency")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Clusters")
        st.components.v1.html(open("analyses/plots/clusters.html", "r").read(), height=500)

    with col2:
    # Display CSV file
        st.markdown("### Cluster Defining Words")
        df = pd.read_csv("analyses/plots/defining_words.csv")

        cluster = st.number_input("Enter the cluster number", value=0)

        # Filter the data based on 'cluster'
        filtered_df = df[df['cluster'] == cluster]

        # Display the filtered data
        st.dataframe(filtered_df)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Popular Tags among Parents")
        st.components.v1.html(open("analyses/plots/parent_popular_tags.html", "r").read(), height=750)
    with col4:
        st.markdown("### Popular Tags among Children")
        st.components.v1.html(open("analyses/plots/child_popular_tags.html", "r").read(), height=500)

    col5, col6 = st.columns(2)
    with col5:
        st.components.v1.html(open("analyses/plots/distribution_flow.html", "r").read(), height=500)
    with col6:
        st.components.v1.html(open("analyses/plots/parent_child_consistency.html", "r").read(), height=500)

    st.markdown("### Closest Families")
    st.components.v1.html(open("analyses/plots/close_families.html", "r").read(), height=500)

    st.markdown("### Parent Child Heatmap")
    st.components.v1.html(open("analyses/plots/parent_child_heatmap.html", "r").read(), height=500)


if __name__ == "__main__":
    main()
