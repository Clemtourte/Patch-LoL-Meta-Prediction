import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="LoL Meta Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0E1117;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
    }
    .prediction-high { color: #00C851; font-weight: bold; }
    .prediction-med { color: #ffbb33; font-weight: bold; }
    .prediction-low { color: #ff4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load real data from your CSV files"""
    try:
        if os.path.exists("error_analysis_by_champion.csv"):
            champion_df = pd.read_csv("error_analysis_by_champion.csv")
            
            # Use your real data columns
            df = champion_df[['champion_name', 'champion_class', 'patch', 'y_true', 'y_pred', 'error', 'abs_error']].copy()
            df = df.rename(columns={
                'champion_name': 'champion',
                'champion_class': 'class',
                'y_true': 'actual_change',
                'y_pred': 'predicted_change',
                'abs_error': 'prediction_error'
            })
            
            # Calculate confidence (inverse of error, normalized)
            max_error = df['prediction_error'].max()
            df['confidence'] = 1 - (df['prediction_error'] / max_error)
            df['confidence'] = df['confidence'].clip(0, 1)
            
            # Get current winrates from your data
            df['current_winrate'] = 50 + df['actual_change'] - df['predicted_change']
            df['predicted_winrate'] = 50 + df['predicted_change']
            
            # Add risk levels
            df['risk_level'] = pd.cut(df['confidence'], bins=[0, 0.7, 0.85, 1.0], 
                                    labels=['High Risk', 'Medium Risk', 'Low Risk'])
            
            # Add simulated pick rates (could be replaced with real data)
            np.random.seed(42)
            df['pick_rate'] = np.random.uniform(5, 25, len(df))
            
            return df
        else:
            st.error("❌ Error analysis data not found. Please run validation scripts first.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_patch_metrics(df, selected_patch):
    """Calculate real metrics for the selected patch vs previous patch"""
    
    # Get patch data
    current_patch_data = df[df['patch'] == selected_patch]
    
    # Get previous patch
    patches = sorted(df['patch'].unique())
    current_idx = patches.index(selected_patch) if selected_patch in patches else -1
    previous_patch = patches[current_idx - 1] if current_idx > 0 else None
    
    metrics = {
        'total_champions': len(current_patch_data),
        'avg_confidence': current_patch_data['confidence'].mean(),
        'buffs_predicted': (current_patch_data['predicted_change'] > 1).sum(),
        'high_impact_changes': (current_patch_data['predicted_change'].abs() > 2).sum(),
        'previous_patch': previous_patch
    }
    
    # Calculate deltas vs previous patch
    if previous_patch:
        previous_patch_data = df[df['patch'] == previous_patch]
        
        metrics['delta_champions'] = len(current_patch_data) - len(previous_patch_data)
        metrics['delta_confidence'] = current_patch_data['confidence'].mean() - previous_patch_data['confidence'].mean()
        metrics['delta_buffs'] = (current_patch_data['predicted_change'] > 1).sum() - (previous_patch_data['predicted_change'] > 1).sum()
        metrics['delta_high_impact'] = (current_patch_data['predicted_change'].abs() > 2).sum() - (previous_patch_data['predicted_change'].abs() > 2).sum()
    else:
        # First patch, no deltas
        metrics['delta_champions'] = 0
        metrics['delta_confidence'] = 0
        metrics['delta_buffs'] = 0
        metrics['delta_high_impact'] = 0
    
    return metrics

@st.cache_data 
def load_historical_data():
    """Load historical validation data"""
    try:
        if os.path.exists("nonconsecutive_validation_results.csv"):
            temporal_df = pd.read_csv("nonconsecutive_validation_results.csv", index_col=0)
            
            historical_data = []
            for idx, row in temporal_df.iterrows():
                historical_data.append({
                    'validation_type': idx,
                    'r2_score': row['r2'],
                    'rmse': row['rmse'],
                    'mae': row['mae']
                })
            
            return pd.DataFrame(historical_data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"⚠️ Could not load historical data: {str(e)}")
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 LoL Meta Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Analytics for Esports Organizations**")
    
    # Load data
    df = load_real_data()
    historical_df = load_historical_data()
    
    if df.empty:
        st.error("❌ No data available. Please run your validation scripts first.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Get available patches
    available_patches = sorted(df['patch'].unique())
    
    selected_patch = st.sidebar.selectbox(
        "📅 Select Patch", 
        available_patches,
        index=len(available_patches)-1 if available_patches else 0
    )
    
    # Calculate metrics for selected patch
    metrics = calculate_patch_metrics(df, selected_patch)
    
    st.sidebar.info(f"📊 **Analyzing patch:** {selected_patch}")
    if metrics['previous_patch']:
        st.sidebar.info(f"📈 **Comparing to:** {metrics['previous_patch']}")
    else:
        st.sidebar.info("📈 **First patch** in dataset")
    
    selected_classes = st.sidebar.multiselect(
        "🎯 Champion Classes",
        df['class'].unique(),
        default=df['class'].unique()
    )
    
    risk_filter = st.sidebar.selectbox(
        "⚠️ Risk Level",
        ["All", "Low Risk", "Medium Risk", "High Risk"],
        index=0
    )
    
    # Filter data for selected patch
    filtered_df = df[
        (df['patch'] == selected_patch) & 
        (df['class'].isin(selected_classes))
    ]
    
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
    
    # Main dashboard layout with REAL metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_text = f"{metrics['delta_champions']:+d} vs {metrics['previous_patch']}" if metrics['previous_patch'] else "First patch"
        st.metric(
            "🎯 Champions Analyzed", 
            metrics['total_champions'],
            delta=delta_text
        )
    
    with col2:
        delta_conf = f"{metrics['delta_confidence']:+.1%}" if metrics['previous_patch'] else "N/A"
        st.metric(
            "🎲 Avg Confidence", 
            f"{metrics['avg_confidence']:.1%}",
            delta=delta_conf
        )
    
    with col3:
        delta_buffs = f"{metrics['delta_buffs']:+d}" if metrics['previous_patch'] else "N/A"
        st.metric(
            "📈 Buffs Predicted", 
            metrics['buffs_predicted'],
            delta=delta_buffs
        )
    
    with col4:
        delta_impact = f"{metrics['delta_high_impact']:+d}" if metrics['previous_patch'] else "N/A"
        st.metric(
            "⚡ High Impact Changes", 
            metrics['high_impact_changes'],
            delta=delta_impact
        )
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "📊 Meta Analysis", "⚠️ Risk Assessment", "🏆 Team Recommendations"])
    
    with tab1:
        st.header("Champion Winrate Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Predictions scatter plot
            fig = px.scatter(
                filtered_df, 
                x='current_winrate', 
                y='predicted_winrate',
                size='pick_rate',
                color='predicted_change',
                hover_data=['champion', 'confidence'],
                color_continuous_scale='RdYlGn',
                title=f"Current vs Predicted Winrates - Patch {selected_patch}"
            )
            fig.add_shape(
                type="line", 
                line=dict(dash="dash", color="white", width=2),
                x0=45, y0=45, x1=55, y1=55
            )
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔥 Biggest Movers")
            
            # Top gainers
            top_gainers = filtered_df.nlargest(3, 'predicted_change')
            st.write("**📈 Predicted Buffs:**")
            for _, row in top_gainers.iterrows():
                confidence_color = "prediction-high" if row['confidence'] > 0.9 else "prediction-med"
                st.markdown(
                    f"<div class='{confidence_color}'>{row['champion']}: +{row['predicted_change']:.1f}%</div>", 
                    unsafe_allow_html=True
                )
            
            st.write("**📉 Predicted Nerfs:**")
            top_losers = filtered_df.nsmallest(3, 'predicted_change')
            for _, row in top_losers.iterrows():
                confidence_color = "prediction-high" if row['confidence'] > 0.9 else "prediction-med"
                st.markdown(
                    f"<div class='{confidence_color}'>{row['champion']}: {row['predicted_change']:.1f}%</div>", 
                    unsafe_allow_html=True
                )
    
    with tab2:
        st.header("Meta Evolution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution analysis
            class_changes = filtered_df.groupby('class')['predicted_change'].mean().reset_index()
            fig = px.bar(
                class_changes, 
                x='class', 
                y='predicted_change',
                color='predicted_change',
                color_continuous_scale='RdYlGn',
                title=f"Average Predicted Change by Class - Patch {selected_patch}"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model performance validation
            if not historical_df.empty:
                fig = px.bar(
                    historical_df, 
                    x='validation_type', 
                    y='r2_score',
                    title="Model Performance Across Validation Scenarios",
                    labels={'r2_score': 'R² Score', 'validation_type': 'Validation Type'}
                )
                fig.update_layout(
                    height=400, 
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Risk Assessment Matrix")
        
        # Risk matrix
        fig = px.scatter(
            filtered_df,
            x='confidence',
            y='predicted_change',
            size='pick_rate',
            color='class',
            hover_data=['champion'],
            title=f"Risk vs Impact Matrix - Patch {selected_patch}"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.add_vline(x=0.85, line_dash="dash", line_color="white")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk table
        st.subheader("📋 Detailed Risk Assessment")
        risk_table = filtered_df[['champion', 'class', 'predicted_change', 'confidence', 'risk_level']].copy()
        risk_table['predicted_change'] = risk_table['predicted_change'].round(1)
        risk_table['confidence'] = (risk_table['confidence'] * 100).round(1)
        st.dataframe(risk_table, use_container_width=True)
    
    with tab4:
        st.header("Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Priority Champions for Practice")
            
            # Champions to prioritize
            priority = filtered_df[
                (filtered_df['predicted_change'] > 1.5) & 
                (filtered_df['confidence'] > 0.85)
            ].sort_values('predicted_change', ascending=False)
            
            if not priority.empty:
                for _, row in priority.head(5).iterrows():
                    st.success(f"🔥 **{row['champion']}** ({row['class']}) - Expected +{row['predicted_change']:.1f}% WR")
            else:
                st.info("No high-priority champions identified for this patch.")
        
        with col2:
            st.subheader("⚠️ Champions to Avoid")
            
            # Champions to avoid
            avoid = filtered_df[
                (filtered_df['predicted_change'] < -1.5) & 
                (filtered_df['confidence'] > 0.85)
            ].sort_values('predicted_change')
            
            if not avoid.empty:
                for _, row in avoid.head(5).iterrows():
                    st.error(f"📉 **{row['champion']}** ({row['class']}) - Expected {row['predicted_change']:.1f}% WR")
            else:
                st.info("No champions flagged as high-risk nerfs.")
        
        st.subheader("📈 Team Composition Recommendations")
        
        # Role-based recommendations
        role_mapping = {
            'Tank': 'Top/Jungle',
            'Fighter': 'Top/Jungle', 
            'Mage': 'Mid',
            'Assassin': 'Mid/Jungle',
            'Marksman': 'ADC',
            'Support': 'Support'
        }
        
        recommendations = {}
        for class_name, role in role_mapping.items():
            class_data = filtered_df[filtered_df['class'] == class_name]
            if not class_data.empty:
                best_pick = class_data.loc[class_data['predicted_change'].idxmax()]
                recommendations[role] = {
                    'champion': best_pick['champion'],
                    'change': best_pick['predicted_change'],
                    'confidence': best_pick['confidence']
                }
        
        for role, data in recommendations.items():
            confidence_icon = "🟢" if data['confidence'] > 0.9 else "🟡" if data['confidence'] > 0.8 else "🔴"
            st.write(f"**{role}**: {data['champion']} {confidence_icon} (+{data['change']:.1f}% predicted)")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        🎮 LoL Meta Predictor | Powered by Advanced ML | Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + f""" | Patch: {selected_patch}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()