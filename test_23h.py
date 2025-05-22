import base64
import datetime
import os
import pandas as pd
import re

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from jinja2 import Environment, FileSystemLoader
import json
from pathlib import Path

from PIL import Image, ImageEnhance

import streamlit as st

from src.custom_classes import OpenAI
from src.fraud_dict import fraud_dictionary

import utility_functions as utils


def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        
        return None

st.set_page_config(page_title="Desjardins ConfIAnce", layout="wide")

st.markdown(
    """
    <style>
    .custom-title {
        color: #00874E !important; /* Change this to your desired color */
        font-size: 3em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="custom-title">Desjarsdins ConfIAnce</h1>', unsafe_allow_html=True)

#st.title("Desjardins ConfIAnce")


st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: #4F5963;
}
</style>
"""
)



# Read the Excel file
df = pd.read_excel('data/age_cat_fraud.xlsx')

# Set the first column as the index (age category)
df = df.set_index(df.columns[0])

# Convert to nested dictionary: {age_category: {fraud_type: value, ...}, ...}
age_fraud_dict = df.to_dict(orient='index')




AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
AZURE_CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]
AZURE_KEYVAULT_NAME = os.environ["AZURE_KEYVAULT_NAME"]

kv_client = SecretClient(
        vault_url=f"https://{AZURE_KEYVAULT_NAME}.vault.azure.net/",
        credential=ClientSecretCredential(
            AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
        ),
    )

llm_conn = OpenAI(kv_client=kv_client)

ASSISTANT_AVATAR = "avatar/alvie32.png"
USER_AVATAR = "❔"




env = Environment(loader=FileSystemLoader('./templates'))
template = env.get_template('assistant_fraude_bancaire.j2')

context = {
    'role': 'Assistant IA spécialisé en détection de fraude bancaire aidant des spécialistes de la fraude à prévenir la fraude et à conseiller efficacement des victimes face aux menaces de plus en plus fréquentes et dangereuses.',
    'objectifs': [
        'Détecter les tentatives de fraude à partir des données historiques et des interactions en temps réel.',
        'Fournir des conseils personnalisés aux victimes potentielles ou avérées.',
        'Soutenir les spécialistes humains dans l’analyse et la décision.',
        'Contribuer à la mise à jour dynamique des bases de connaissances sur la fraude.'
    ],
    'directives': [
        'Offrir des réponses factuelles et contextualisées.',
        'Respecter les lois sur la protection des données (ex. : Loi 25).',
        'Préconiser des actions de prévention et d’escalade en cas de doute.'
    ],
    'contraintes': [
        'Ne jamais divulguer ou générer de données personnelles réelles.',
        'Ne pas conclure de manière définitive sans confirmation humaine.',
        'Éviter toute spéculation non fondée.'
    ],
    'ton': 'professionnel, rassurant et pédagogique',
    'public': 'des experts en sécurité, des conseillers bancaires et des clients ayant un niveau de connaissance variable',
    'principes': [
        'Baser toute recommandation sur des preuves ou tendances documentées.',
        'Être transparent sur les limites de l’analyse automatique.',
        'Préférer une approche préventive et explicative.'
    ]
}

system_message = template.render(context)
#print(system_message)





# Sidebar (collapsible)
with st.sidebar:
    st.title("À propos")
    st.write("Détectez la fraude, évaluez votre situation et obtenez des solutions adaptées.")
    st.markdown("**References:**")
    #st.markdown("- [Fraud Prevention Tips](#)")
    st.markdown(
    '- <a href="https://antifraudcentre-centreantifraude.ca/index-fra.htm" style="color: #8EC9A9; text-decoration: underline;" target="_blank">Centre Anti-Fraude Canadien</a>',
    unsafe_allow_html=True)
    st.markdown(
    '- <a href="https://www.desjardins.com/securite/index.jsp" style="color: #8EC9A9; text-decoration: underline;" target="_blank">Desjardins Sécurité</a>',
    unsafe_allow_html=True)
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


# Top right: Contact button
st.markdown(
    """
    <div style='position: absolute; top: 10px; right: 20px;'>
        <a href='tel:1-800-224-7737'>
            <button style='background-color:#00874E; color:white; padding:10px 20px; border:none; border-radius:5px; font-size:16px;'>
                Appelez-nous
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

#st.title("Fraud Awareness Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "Vous êtes un assistant de prévention contre la fraude."
            "Posez des questions étape par étape à l'utilisateur afin de comprendre sa situation."
            "Une fois que vous avez suffisamment d'informations, Vous devez ABSOLUMENT débuter écrire 'ÉVALUATION COMPLÉTÉE' "
            "Ensuite, évaluez la probabilité de fraude, identifiez le type de fraude (à partir du dictionnaire), décrivez-le et suggérez les prochaines étapes."
            "Voici le dictionnaire des fraudes à titre de référence :\n"
            f"{fraud_dictionary}\n"
            "Commencez par saluer l'utilisateur et poser votre première question."
        )},
        {"role": "assistant", "content": (
            "Bonjour, je suis ici pour vous aider à examiner toute situation qui vous semble inhabituelle ou suspecte. \n\n"
            "N'hésitez pas à me décrire ce qui s'est passé, même si cela vous paraît anodin. \n\n"
            "Ensemble, nous allons y voir plus clair et déterminer s’il pourrait s’agir d’une tentative de fraude. \n\n"
            "Que s’est-il passé exactement ?"
        )}
    ]


if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False

tab1, tab2, tab3 = st.tabs(["Chat avec ConfIAnce", "ConfIAnce Insights", "À propos"])

# Place your existing chat interface in the first tab
with tab1:
    # Display chat history with custom avatars
    for i, message in enumerate(st.session_state.messages):
        if message["role"] != "system":
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                    st.markdown(message["content"])
                    # Only allow feedback on the last assistant response, and not on the first assistant message
                    if i == len(st.session_state.messages) - 1 and i > 1:
                        left_col, right_col = st.columns([4, 1])
                        with right_col:
                            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
                            selected = st.feedback("thumbs", key=f"feedback_{i}")
                            if selected is not None:
                                st.caption("_Merci de la rétroaction._")
                                # save user feedback to a json file
                                # Get the last two messages (user and assistant)
                                user_feedback = "positive" if selected == 1 else "negative"
                                last_two_messages = st.session_state.messages[-2:]
                                feedback_dict = {
                                    "user_message": last_two_messages[0]["content"],
                                    "assistant_message": last_two_messages[1]["content"],
                                    "feedback": user_feedback
                                }
                                # Save feedback...
                                ROOT_DIR = Path.cwd()
                                # Get the root directory (assuming this script is in src/)
                                project_root = Path(ROOT_DIR).resolve()#.parents[0] 
                                # Access the data file
                                data_path = project_root / 'user_feedback'
                                feedback_file_path = data_path / 'user_feedback.json'

                                with open(feedback_file_path, "a", encoding="utf-8") as f:
                                    # read last line of json file
                                    try:
                                        with open(feedback_file_path, "r", encoding="utf-8") as f:
                                            last_line = f.readlines()[-1]
                                    except Exception as e:
                                        print(f"Error reading file: {e}")
                                
                                # modify last line if the user message and assistant message are the same, else append new line
                                # check if the last line is the same as the current feedback
                                try:
                                    import ast
                                    last_line_dict = ast.literal_eval(last_line) if last_line else None
                                except Exception as e:
                                    last_line_dict = None

                                if last_line_dict and feedback_dict.get("user_message") == last_line_dict.get("user_message") and feedback_dict.get("assistant_message") == last_line_dict.get("assistant_message"):
                                    # Overwrite the last line with the new feedback
                                    with open(feedback_file_path, "r", encoding="utf-8") as f:
                                        lines = f.readlines()
                                    lines[-1] = json.dumps(feedback_dict) + "\n"
                                    with open(feedback_file_path, "w", encoding="utf-8") as f:
                                        f.writelines(lines)
                                else:
                                    # Append new feedback
                                    with open(feedback_file_path, "a", encoding="utf-8") as f:
                                        json.dump(feedback_dict, f)
                                        f.write("\n")

            else:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(message["content"])

    # Only allow new input if assessment is not complete
    if not st.session_state.assessment_complete:
        user_input = st.chat_input("Votre réponse")

        # Check if user input is not empty
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            context = st.session_state.messages[-10:]  # last 10 messages for brevity
            
            # check if user input has a scam email, phone or url provided
            scam_email_found, scam_email = utils.identify_scam_email(user_input)
            scam_phone_found, scam_phone = utils.identify_scam_phone(user_input)

            if st.session_state.get('has_scam_email', False):
                st.warning(f"⚠️ Le courriel {st.session_state.scam_email_value} est un fraudeur confirmé. Veuillez cesser tout contact et suivre les conseils suivants.")


            if scam_email_found:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Le courriel {scam_email} provient d'une liste suspecte de fraude."})
                st.session_state.messages.append({"role": "system", "content": "Le courriel provient d'une liste suspecte de fraude. Donner des conseils immédiatement en se basant sur les directives du dictionnaire de fraude."})
                st.warning(f"⚠️ Le courriel {scam_email} est un fraudeur confirmé. Veuillez cesser tout contact et suivre les conseils suivants.")

            if scam_phone_found:
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Le numéro de téléphone {scam_phone} provient d'une liste suspecte de fraude."})
                st.session_state.messages.append({"role": "system", "content": "Le numéro de téléphone provient d'une liste suspecte de fraude. Donner des conseils immédiatement en se basant sur les directives du dictionnaire de fraude."})
                # st.warning(f"⚠️ Le numéro de téléphone {scam_phone} est un fraudeur confirmé. Veuillez cesser tout contact et suivre les conseils suivants.")


    
    # Prepare messages for API (custom format required)
            api_messages = {
                "messages": [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                    # if msg["role"] != "system"  # Include system prompt only once
                ]
            }
            api_messages["messages"].insert(0, st.session_state.messages[0])  # Add system prompt

        # Get LLM response using custom method
            #print(api_messages)
            bot_response = llm_conn.generic_completion(messages=api_messages)

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            def contains_any_keyword(s, keywords):
                # Join keywords into a single regex pattern, using word boundaries for whole word matches
                print("checking")
                pattern = r'|'.join(re.escape(word) for word in keywords)
                return bool(re.search(pattern, s, re.IGNORECASE))

            keywords = ["Probabilité de fraude",
                        "Signaler la fraude", 
                        "ÉVALUATION COMPLÉTÉE",
                        "D'après les informations fournies"]
            result = contains_any_keyword(bot_response, keywords)

            if result == True:
                print("detected keyword to complete the assessment")
                st.session_state.assessment_complete = True
                #st.session_state.messages.append({"role": "system", "content": (
                #    "Vous êtes un assistant de prévention contre la fraude."
                #    "Vous venez de terminer votre évaluation et avez identifier le type de fraude dans cette situation spécifique"
                #    "Avec le type de fraude identifé, utilisez la table statistique suivante pour illustrer les tranches d'âge ciblées:\n"
                #    f"{age_fraud_dict}\n"
                #    "Donnez immédiatement de l'information statistique"
                #)})
            st.rerun()
    else:
        st.info("Merci d'avoir discuter avec ConfIAnce. N'hésitez pas à nous recontacter.")

with tab2:
    st.header("ConfIAnce Insights")
    st.subheader("Tendances émergentes en matière de fraude")
    
    # Import data
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        
        @st.cache_data
        def load_fraud_data():
            ROOT_DIR = Path.cwd()

            # Get the root directory (assuming this script is in src/)
            project_root = Path(ROOT_DIR).resolve()#.parents[0] 

            # Access the data file
            data_path = project_root / 'data'
            data_path = data_path / 'fraude_type_desc_conseil.xlsx'

            return pd.read_excel(data_path)
        
        # Load data
        with st.spinner("Chargement des données..."):
            df = load_fraud_data()
            
        # Display an overview of the dataset
        st.write("### Aperçu du dictionnaire de fraude")
        st.dataframe(df.head(10))
        
        # Display basic metrics
        st.write("### Statistiques de base")
        st.metric("Nombre total d'entrées", f"{len(df):,}")
        st.metric("Types de fraudes distincts", f"{df['Fraude_Nom'].nunique():,}")
        
        # Bar chart of fraud occurrences
        st.write("### Répartition des fraudes")
        fraud_counts = df['Fraude_Nom'].value_counts().reset_index()
        fraud_counts.columns = ["Fraude_Nom", "Occurrences"]
        
        fig = px.bar(
            fraud_counts,
            x="Fraude_Nom",
            y="Occurrences",
            title="Nombre de cas par type de fraude",
            color="Fraude_Nom",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key observations in a fancy text block
        st.write("### Observations clés")
        st.info("""
        **Tendances notables identifiées dans les données :**
        
        1. **Diversité des types de fraude :** Le dictionnaire répertorie un grand nombre de fraudes diverses, allant des extorsions aux manipulations financières.
        2. **Focalisation sur la prévention :** Chaque type de fraude est accompagné de conseils ciblés, permettant d’améliorer la prévention.
        3. **Importance de la vigilance :** Des indicateurs visuels tels que la répartition des fraudes et les nuages de mots démontrent l’importance de prêter attention aux tendances émergentes.
        """)

    except Exception as e:
        st.error(f"Erreur lors de l'analyse des données: {e}")
        st.info("Assurez-vous que le fichier fraude_type_desc_conseil.xlsx existe dans le dossier ./data")

with tab3:
    st.write("À propos de cette application")
    st.markdown(
        """
        Cette application utilise l'intelligence artificielle pour aider à détecter et prévenir la fraude bancaire. 
        Elle est conçue pour interagir avec les utilisateurs, poser des questions et fournir des conseils basés sur les informations fournies.
        """
    )
    st.markdown(
        """
        **Fonctionnalités :**
        - Détection de la fraude
        - Conseils personnalisés
        - Support aux spécialistes humains (à venir)
        """
    )
    st.markdown(
        """
        **Technologies utilisées :**
        - Azure OpenAI Service
        - Streamlit
        - Jinja2 pour le rendu de templates
        """
    )
    st.markdown(
        """
        **Sources de données utilisées :**
        - https://www.fraude-alerte.ca/scam/view/550079#comment-1290533
        - https://antifraudcentre-centreantifraude.ca/index-fra.htm
        - Jinja2 pour le rendu de templates
        """
    )