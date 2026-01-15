
"""
Streamlit application for Monte Carlo Risk Simulation with ML and GenAI insights.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from ml import (
    run_simulation, 
    calculate_drawdown, 
    apply_scenario, 
    train_ml_surrogate, 
    predict_var_ml,
    calculate_risk_metrics
)
from genai import generate_risk_report, get_quick_insight

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Risk Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Monte Carlo Risk Simulator</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Advanced Risk Analysis for SMEs & Startups</p>', unsafe_allow_html=True)


# Sidebar Inputs
st.sidebar.header("📊 Input Parameters")

st.sidebar.markdown("### Capital & Cash Flow")
initial_capital = st.sidebar.number_input(
    "Initial Capital", 
    value=100000, 
    step=5000,
    help="Starting amount of capital available"
)

daily_inflow = st.sidebar.number_input(
    "Daily Inflow ", 
    value=500, 
    step=50,
    help="Average daily revenue or cash inflow"
)

daily_outflow = st.sidebar.number_input(
    "Daily Outflow ", 
    value=600, 
    step=50,
    help="Average daily expenses or cash outflow"
)

st.sidebar.markdown("### Market Conditions")
volatility = st.sidebar.number_input(
    "Volatility (std dev)", 
    value=0.01, 
    min_value=0.001,
    max_value=0.1,
    step=0.001, 
    format="%.3f",
    help="Market volatility - higher means more unpredictable"
)

st.sidebar.markdown("### Simulation Settings")
scenario = st.sidebar.selectbox(
    "Scenario", 
    ["Normal", "High Volatility", "High Burn"],
    help="Test different risk scenarios"
)

n_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
    help="More simulations = more accurate but slower"
)

# Apply scenario modifications
initial_capital_mod, daily_inflow_mod, daily_outflow_mod, volatility_mod = apply_scenario(
    initial_capital, daily_inflow, daily_outflow, volatility, scenario
)

# Show modified parameters if scenario changed them
if scenario != "Normal":
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **📌 Scenario Adjustments:**
    - Volatility: {volatility_mod:.3f}
    - Daily Outflow: {daily_outflow_mod:.2f}
    """)

# Calculate and display burn rate info
net_burn = daily_outflow_mod - daily_inflow_mod
runway_days = initial_capital_mod / net_burn if net_burn > 0 else float('inf')

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
if net_burn > 0:
    st.sidebar.error(f"⚠️ Daily Net Burn: {net_burn:.2f}")
    st.sidebar.warning(f"🕐 Runway: ~{runway_days:.0f} days ({runway_days/30:.1f} months)")
else:
    st.sidebar.success(f"✅ Daily Net Gain: {-net_burn:.2f}")

# -------------------------------
# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Run Simulation")
    st.markdown("Click the button below to run the Monte Carlo simulation and analyze your financial risk.")

with col2:
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# -------------------------------
# Run Monte Carlo simulation
if run_button:
    # Create tabs for organized output
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visualizations", "🤖 ML Model", "💡 AI Insights"])
    
    with tab1:
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulation
            all_paths, VaR_95 = run_simulation(
                initial_capital_mod, 
                daily_inflow_mod, 
                daily_outflow_mod, 
                volatility_mod,
                n_simulations=n_simulations
            )
            
            # Calculate additional metrics
            drawdowns = calculate_drawdown(all_paths)
            max_drawdown = max(drawdowns)
            avg_drawdown = np.mean(drawdowns)
            risk_metrics = calculate_risk_metrics(all_paths)
        
        st.success("✅ Simulation completed!")
        
        # Display key metrics in cards
        st.markdown("### 📋 Key Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_pct = ((initial_capital_mod - VaR_95) / initial_capital_mod) * 100
            st.metric(
                "VaR 95%", 
                f"{VaR_95:,.0f}",
                delta=f"-{risk_pct:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Max Drawdown", 
                f"{max_drawdown*100:.2f}%",
                delta=f"Avg: {avg_drawdown*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Final Capital", 
                f"{risk_metrics['mean']:,.0f}",
                delta=f"+{((risk_metrics['mean']-initial_capital_mod)/initial_capital_mod)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Prob. of Loss",
                f"{risk_metrics['prob_loss']:.1f}%"
            )
        
        # Additional metrics in expandable section
        with st.expander("📊 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribution Metrics:**")
                st.write(f"- Median Final Capital: {risk_metrics['median']:,.2f}")
                st.write(f"- Standard Deviation: {risk_metrics['std']:,.2f}")
                st.write(f"- Best Case (Max): {risk_metrics['best_case']:,.2f}")
                st.write(f"- Worst Case (Min): {risk_metrics['worst_case']:,.2f}")
            
            with col2:
                st.markdown("**Risk Measures:**")
                st.write(f"- VaR 95%: {risk_metrics['var_95']:,.2f}")
                st.write(f"- VaR 99%: {risk_metrics['var_99']:,.2f}")
                st.write(f"- Average Drawdown: {avg_drawdown*100:.2f}%")
                st.write(f"- Max Drawdown: {max_drawdown*100:.2f}%")
        
        # Quick insight without API
        quick_insight = get_quick_insight(VaR_95, initial_capital_mod, max_drawdown)
        st.info(quick_insight)
    
    with tab2:
        st.markdown("### 📈 Monte Carlo Simulation Paths")
        
        # Create Monte Carlo paths visualization
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        
        n_plot = min(100, all_paths.shape[0])
        
        # Plot individual paths
        for i in range(n_plot):
            ax1.plot(all_paths[i], alpha=0.2, linewidth=0.8, color='steelblue')
        
        # Add mean path
        mean_path = np.mean(all_paths, axis=0)
        ax1.plot(mean_path, color='darkblue', linewidth=3, label='Mean Path', zorder=5)
        
        # Add percentile bands
        p25 = np.percentile(all_paths, 25, axis=0)
        p75 = np.percentile(all_paths, 75, axis=0)
        ax1.fill_between(range(len(p25)), p25, p75, alpha=0.2, color='green', label='25th-75th Percentile')
        
        # Add VaR line
        ax1.axhline(y=VaR_95, color='red', linestyle='--', linewidth=2.5, label=f'VaR 95%: {VaR_95:,.0f}', zorder=5)
        
        # Add initial capital line
        ax1.axhline(y=initial_capital_mod, color='gray', linestyle=':', linewidth=2, label=f'Initial: {initial_capital_mod:,.0f}', zorder=5)
        
        ax1.set_xlabel("Time Steps (Days)", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Capital", fontsize=13, fontweight='bold')
        ax1.set_title(f"Monte Carlo Capital Paths - {scenario} Scenario ({n_simulations} simulations)", 
                     fontsize=15, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        st.pyplot(fig1)
        plt.close(fig1)
        
        # Drawdown Distribution
        st.markdown("### 📉 Drawdown Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        ax2.hist(drawdowns, bins=40, color='coral', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.axvline(x=max_drawdown, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Max DD: {max_drawdown*100:.2f}%', zorder=5)
        ax2.axvline(x=avg_drawdown, color='orange', linestyle='--', linewidth=2.5, 
                   label=f'Avg DD: {avg_drawdown*100:.2f}%', zorder=5)
        
        ax2.set_xlabel("Drawdown (%)", fontsize=13, fontweight='bold')
        ax2.set_ylabel("Frequency", fontsize=13, fontweight='bold')
        ax2.set_title("Distribution of Maximum Drawdowns", fontsize=15, fontweight='bold', pad=20)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
        
        st.pyplot(fig2)
        plt.close(fig2)
        
        # Final capital distribution
        st.markdown("### 📊 Final Capital Distribution")
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        final_capitals = all_paths[:, -1]
        ax3.hist(final_capitals, bins=50, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax3.axvline(x=VaR_95, color='red', linestyle='--', linewidth=2.5, 
                   label=f'VaR 95%: {VaR_95:,.0f}', zorder=5)
        ax3.axvline(x=risk_metrics['mean'], color='green', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {risk_metrics["mean"]:,.0f}', zorder=5)
        ax3.axvline(x=initial_capital_mod, color='gray', linestyle=':', linewidth=2, 
                   label=f'Initial: {initial_capital_mod:,.0f}', zorder=5)
        
        ax3.set_xlabel("Final Capital", fontsize=13, fontweight='bold')
        ax3.set_ylabel("Frequency", fontsize=13, fontweight='bold')
        ax3.set_title("Distribution of Final Capital Values", fontsize=15, fontweight='bold', pad=20)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        st.pyplot(fig3)
        plt.close(fig3)
    
    with tab3:
        st.markdown("### 🤖 Machine Learning Surrogate Model")
        st.markdown("Training a Random Forest model to predict VaR instantly without full simulation...")
        
        with st.spinner("Training ML model..."):
            # Train ML surrogate
            rf_model, mse = train_ml_surrogate(
                initial_capital_mod, 
                daily_inflow_mod, 
                daily_outflow_mod, 
                volatility_mod, 
                n_simulations=min(300, n_simulations)
            )
            
            # Get ML prediction
            predicted_var = predict_var_ml(
                rf_model, 
                initial_capital_mod, 
                daily_inflow_mod, 
                daily_outflow_mod, 
                volatility_mod
            )
        
        st.success("✅ ML model trained successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model MSE", f"{mse:,.2f}")
        
        with col2:
            st.metric("ML Predicted VaR 95%", f"{predicted_var:,.0f}")
        
        with col3:
            error_pct = abs(predicted_var - VaR_95) / VaR_95 * 100
            st.metric("Prediction Error", f"{error_pct:.2f}%")
        
        # Comparison chart
        st.markdown("#### 🔄 Monte Carlo vs ML Prediction Comparison")
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        methods = ['Monte Carlo\nSimulation', 'ML Surrogate\nModel']
        values = [VaR_95, predicted_var]
        colors = ['steelblue', 'coral']
        
        bars = ax4.bar(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4.set_ylabel("VaR 95%", fontsize=13, fontweight='bold')
        ax4.set_title("Comparison: Monte Carlo vs ML Prediction", fontsize=15, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        st.pyplot(fig4)
        plt.close(fig4)
        
        st.info(f"""
        **How it works:** The ML model learns patterns from simulation data and can predict VaR instantly.
        - **Accuracy:** {100-error_pct:.1f}% match with Monte Carlo
        - **Speed:** ~1000x faster than full simulation
        - **Use case:** Rapid scenario testing and real-time risk assessment
        """)
    
    with tab4:
        st.markdown("### 💡 AI-Powered Risk Analysis")
        
        enable_genai = st.checkbox("🧠 Generate AI Insights", value=True, 
                                   help="Uses OpenAI GPT-4o-mini to analyze your risk profile")
        
        if enable_genai:
            with st.spinner("Generating AI-powered insights... This may take 10-15 seconds."):
                try:
                    insight = generate_risk_report(
                        VaR_95, 
                        initial_capital_mod, 
                        daily_inflow_mod, 
                        daily_outflow_mod, 
                        volatility_mod
                    )
                    
                    st.markdown("#### 📝 Risk Analysis Report")
                    st.markdown(insight)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=insight,
                        file_name="risk_analysis_report.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating AI insight: {str(e)}")
                    
        

# -------------------------------
# Information sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    This simulator uses:
    - **Monte Carlo**: Generates probabilistic scenarios
    - **VaR 95%**: Measures worst-case risk
    - **Drawdown**: Tracks capital decline
    - **ML Model**: Fast risk prediction
    - **AI Analysis**: Actionable insights
    
    **Scenarios:**
    - 🟢 Normal: Standard conditions
    - 🟡 High Volatility: 2x market uncertainty
    - 🔴 High Burn: 1.5x spending rate
    """)
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Start with Normal scenario, then test stress scenarios.")