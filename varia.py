

st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: #383838;
}
</style>
"""
)

# Sidebar (collapsible)
with st.sidebar:
    st.title("À propos")
    st.write("Détectez la fraude, évaluez votre situation et obtenez des solutions adaptées.")
    st.markdown("**References:**")
    #st.markdown("- [Fraud Prevention Tips](#)")
    st.markdown("- [Centre Anti-Fraude Canadien](https://antifraudcentre-centreantifraude.ca/index-fra.htm)")
    st.markdown("- [Desjardins Sécurité](https://www.desjardins.com/securite/index.jsp)")
    st.write(f"**Date:** {datetime.date.today()}")
    st.markdown("---")
    st.caption("Solutionné avec ❤️ par les Hackstreet Boys")


        # Load and display image with glowing effect
    img_path = "avatar/hackathon2025_small.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<div style="text-align: center;"><img src="data:image/png;base64,{img_base64}" class="cover-glow"> </div>',
            unsafe_allow_html=True,
        )
