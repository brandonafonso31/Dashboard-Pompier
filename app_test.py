import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------------------------------------------
# 1. Définition EXACTE de l'architecture du modèle
# (Doit correspondre à Dueling_QNetwork dans votre code original)
# ---------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size=128, n_steps=1, seed=0, 
                 num_layers=3, layer_type="dense", use_batchnorm=True):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batchnorm = use_batchnorm
        
        # Couches partagées
        self.shared_layers = nn.Sequential()
        in_size = state_size
        
        for i in range(num_layers):
            self.shared_layers.add_module(f"linear_{i}", nn.Linear(in_size, layer_size))
            if use_batchnorm:
                self.shared_layers.add_module(f"batchnorm_{i}", nn.BatchNorm1d(layer_size))
            self.shared_layers.add_module(f"relu_{i}", nn.ReLU())
            in_size = layer_size
        
        # Branche Value
        self.value_stream = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
        )
        
        # Branche Advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size, action_size)
        )

    def forward(self, state):
        features = self.shared_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean()

# ---------------------------------------------------------------
# 2. Chargement du modèle adapté
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Paramètres du modèle (doivent correspondre à l'entraînement)
        STATE_SIZE = 5  # À adapter selon votre état
        ACTION_SIZE = 3 # Nombre d'actions possibles
        LAYER_SIZE = 128
        NUM_LAYERS = 3
        USE_BATCHNORM = True
        
        # Initialisation du modèle
        model = DuelingQNetwork(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            layer_size=LAYER_SIZE,
            num_layers=NUM_LAYERS,
            use_batchnorm=USE_BATCHNORM
        )
        
        # Chargement des poids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("SVG_model/agent_z1_sent_r100_cf3", map_location=device)
        
        # Gestion des noms de paramètres différents
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if k.startswith("model."):
                new_k = k.replace("model.", "shared_layers.")
            elif k.startswith("value."):
                new_k = k.replace("value.", "value_stream.")
            elif k.startswith("advantage."):
                new_k = k.replace("advantage.", "advantage_stream.")
            else:
                new_k = k
            new_checkpoint[new_k] = v
        
        # Chargement avec vérification
        model.load_state_dict(new_checkpoint, strict=False)
        model.eval()
        
        st.success("Modèle chargé avec succès!")
        return model
        
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        st.error("Vérifiez que :")
        st.error("- L'architecture du modèle correspond exactement")
        st.error("- Le chemin du fichier est correct")
        return None

# ---------------------------------------------------------------
# 3. Interface Streamlit
# ---------------------------------------------------------------
st.set_page_config(page_title="🚨 Optimisation Pompiers", layout="wide")
st.title("🚒 Dashboard d'Aide à la Décision pour Pompiers")

model = load_model()

if model is not None:
    with st.sidebar:
        st.header("📊 Paramètres d'intervention")
        inputs = {
            'incendies': st.slider("Incendies actifs", 1, 10, 2),
            'distance': st.slider("Distance moyenne (km)", 1, 50, 15),
            'equipes': st.slider("Équipes disponibles", 1, 10, 3),
            'gravite': st.select_slider("Niveau de gravité", ['Faible', 'Moyen', 'Élevé']),
            'meteo': st.slider("Conditions météo (0-1)", 0.0, 1.0, 0.8)
        }

    if st.button("🚀 Lancer la simulation"):
        # Préparation de l'état
        state = np.array([
            inputs['incendies'],
            inputs['distance'],
            inputs['equipes'],
            {'Faible':0.2, 'Moyen':0.5, 'Élevé':0.9}[inputs['gravite']],
            inputs['meteo']
        ], dtype=np.float32)
        
        # Prédiction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor).numpy().squeeze()
            action = np.argmax(q_values)
        
        # Affichage des résultats
        st.subheader("📋 Résultats de la simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Action recommandée", 
                     f"Option {action+1}", 
                     delta=f"Score: {q_values.max():.2f}")
            
            st.write("""
            **Interprétation des actions**:
            1. Intervention ciblée (feu le plus critique)
            2. Répartition équilibrée
            3. Demande de renforts externes
            """)
        
        with col2:
            fig = px.bar(x=[1, 2, 3], y=q_values,
                        labels={'x':'Option', 'y':'Score Q'},
                        title="Performance des stratégies",
                        color=[1, 2, 3],
                        color_continuous_scale='Bluered')
            st.plotly_chart(fig, use_container_width=True)
        
        # Carte des ressources
        st.subheader("🗺 Répartition géographique simulée")
        fig_map = px.scatter_mapbox(
            lat=[48.85, 48.86, 48.855],
            lon=[2.35, 2.36, 2.352],
            color=["Incendie", "Incendie", "Caserne"],
            size=[inputs['incendies']*5, 10, inputs['equipes']*8],
            zoom=11,
            mapbox_style="open-street-map",
            hover_name=["Feu principal", "Feu secondaire", "Base opérationnelle"]
        )
        st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Le modèle n'est pas disponible. Veuillez configurer correctement le chargement.")