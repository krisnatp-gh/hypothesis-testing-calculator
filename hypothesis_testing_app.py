import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import t as t_dist

# Page configuration
st.set_page_config(
    page_title="Hypothesis Testing: t-Test",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
#st.markdown("""
#<style>
#    .stApp {
#        background-color: #0f172a;
#    }
#    .stButton>button {
#        background-color: #1e293b;
#        color: #06b6d4;
#        border: 2px solid #06b6d4;
#        border-radius: 4px;
#        font-weight: 600;
#    }
#    .stButton>button:hover {
#        background-color: #06b6d4;
#        color: #0f172a;
#    }
#    div[data-testid="stNumberInput"] label {
#        color: #9ca3af;
#        font-weight: 500;
#    }
#    div[data-baseweb="radio"] label {
#        color: #9ca3af;
#    }
#</style>
#""", unsafe_allow_html=True)

# Title
st.title("Interactive Hypothesis Testing: One-Sample t-Test")
# st.markdown("Learn hypothesis testing intuitions by exploring how test statistics relate to decisions")

st.markdown("---")

# Initialize session state for alpha
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.05

def set_alpha(value):
    """Callback function to update alpha in session state"""
    st.session_state.alpha = value

# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_standard_error(s, n):
    """Calculate standard error"""
    return s / np.sqrt(n)

def calculate_t_statistic(x_bar, mu_0, se):
    """Calculate t-statistic"""
    return (x_bar - mu_0) / se

def calculate_p_value(t_stat, df, test_type):
    """Calculate p-value based on test type"""
    if test_type == "Two-tailed":
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    elif test_type == "Left-tailed":
        p_value = t_dist.cdf(t_stat, df)
    else:  # Right-tailed
        p_value = 1 - t_dist.cdf(t_stat, df)
    return p_value

# ============================================================================
# SECTION 1: TEST INPUTS, TEST TYPE SELECTION, AND CALCULATIONS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    # Enter Test Statistics
    st.markdown("#### Enter Test Statistics and Theoretical Population Mean")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown("#### Sample Mean (x̄)")
        x_bar = st.number_input("Sample Mean (x̄)", value=95.7, step=1.0, format="%.2f", label_visibility="collapsed")

        st.markdown("#### Sample Standard Deviation (s)")
        s = st.number_input("Sample Standard Deviation (s)", value=20.0, min_value=0.5, step=0.5, format="%.2f", label_visibility="collapsed")

    with input_col2:
        st.markdown("#### Theoretical population mean (μ₀)")
        mu_0 = st.number_input("Theoretical population mean (μ₀)", value=100.0, format="%.2f", label_visibility="collapsed")

        st.markdown("#### Number of samples (n)")
        n = st.number_input("Number of samples (n)", value=100, min_value=50, step=1, label_visibility="collapsed")

    st.markdown("")  # Add spacing

    # Select Test Type
    st.markdown("#### Select Test Type")

    # Create nested columns for radio buttons and hypotheses
    nested_col1, nested_col2 = st.columns(2)

    with nested_col1:
        test_type = st.radio(
            "Select the alternative hypothesis:",
            ["Two-tailed",
             "Left-tailed",
             "Right-tailed"],
            horizontal=False,
            label_visibility="collapsed"
        )

    with nested_col2:
        # Extract test type key and display hypotheses using LaTeX
        if "Two-tailed" in test_type:
            test_key = "Two-tailed"
            st.latex(r"H_0: \mu = \mu_0")
            st.latex(r"H_A: \mu \neq \mu_0")
        elif "Left-tailed" in test_type:
            test_key = "Left-tailed"
            st.latex(r"H_0: \mu = \mu_0")
            st.latex(r"H_A: \mu < \mu_0")
        else:
            test_key = "Right-tailed"
            st.latex(r"H_0: \mu = \mu_0")
            st.latex(r"H_A: \mu > \mu_0")

with col2:
    st.markdown("#### t-statistic Calculation")

    # Perform calculations
    df = n - 1
    se = calculate_standard_error(s, n)
    t_stat = calculate_t_statistic(x_bar, mu_0, se)
    p_value = calculate_p_value(t_stat, df, test_key)
    decision = "Reject H₀" if p_value < st.session_state.alpha else "Fail to reject H₀"

    with st.expander("Show Calculations", expanded=True):
        # Standard Error Display
        st.markdown("##### Standard Error")
        st.latex(rf"SE = \frac{{s}}{{\sqrt{{n}}}} = \frac{{{s:.4f}}}{{\sqrt{{{n}}}}} = {se:.4f}")

        # st.markdown("")

        # T-statistic Display
        st.markdown("##### t-Statistic")
        st.latex(rf"t = \frac{{\bar{{x}} - \mu_0}}{{SE}} = \frac{{{x_bar} - {mu_0}}}{{{se:.4f}}} = {t_stat:.4f}")

# st.markdown("---")

# ============================================================================
# SECTION 2: VISUALIZATION
# ============================================================================

def create_t_distribution_plot(df, alpha, t_stat, test_type):
    """Create t-distribution visualization with shaded confidence interval"""

    # Generate t-distribution curve (clipped at 0.1% and 99.9% percentiles)
    x_min, x_max = t_dist.ppf(0.001, df), t_dist.ppf(0.999, df)
    x = np.linspace(x_min, x_max, 1000)
    y = t_dist.pdf(x, df)

    fig = go.Figure()

    # Determine critical values and shading based on test type
    if test_type == "Two-tailed":
        t_crit_lower = t_dist.ppf(alpha/2, df)
        t_crit_upper = t_dist.ppf(1 - alpha/2, df)

        # Shade middle (1-α) confidence region in cyan
        confidence_x = x[(x >= t_crit_lower) & (x <= t_crit_upper)]
        confidence_y = t_dist.pdf(confidence_x, df)

        fig.add_trace(go.Scatter(
            x=confidence_x, y=confidence_y,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.25)',
            line=dict(width=0),
            name=f'{(1-alpha)*100:.0f}% Confidence Region',
            hoverinfo='skip',
            showlegend=True
        ))

        # Add critical value annotations
        # fig.add_vline(x=t_crit_lower, line_dash="dot", line_color="#146fef", line_width=2,
        #               annotation_text=f"<b>t* = {t_crit_lower:.2f}</b>", annotation_position="bottom left")
        # fig.add_vline(x=t_crit_upper, line_dash="dot", line_color="#146fef", line_width=2,
        #               annotation_text=f"<b>t* = {t_crit_upper:.2f}</b>", annotation_position="bottom right")

    elif test_type == "Left-tailed":
        t_crit = t_dist.ppf(alpha, df)

        # Shade right side (1-α) confidence region
        confidence_x = x[x >= t_crit]
        confidence_y = t_dist.pdf(confidence_x, df)

        fig.add_trace(go.Scatter(
            x=confidence_x, y=confidence_y,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.25)',
            line=dict(width=0),
            name=f'{(1-alpha)*100:.0f}% Confidence Region',
            hoverinfo='skip',
            showlegend=True
        ))

        # # Add critical value annotation
        # fig.add_vline(x=t_crit, line_dash="dot", line_color="#4b5563", line_width=2,
        #               annotation_text=f"<b>t* = {t_crit:.2f}</b>", annotation_position="bottom right")

    else:  # Right-tailed
        t_crit = t_dist.ppf(1 - alpha, df)

        # Shade left side (1-α) confidence region
        confidence_x = x[x <= t_crit]
        confidence_y = t_dist.pdf(confidence_x, df)

        fig.add_trace(go.Scatter(
            x=confidence_x, y=confidence_y,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.25)',
            line=dict(width=0),
            name=f'{(1-alpha)*100:.0f}% Confidence Region',
            hoverinfo='skip',
            showlegend=True
        ))

        # # Add critical value annotation
        # fig.add_vline(x=t_crit, line_dash="dot", line_color="#4b5563", line_width=2,
        #               annotation_text=f"<b>t* = {t_crit:.2f}</b>", annotation_position="bottom left")

    # Main distribution curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#0369a1', width=4),
        name='t-distribution',
        hovertemplate='t = %{x:.3f}<br>Density = %{y:.4f}<extra></extra>'
    ))

    # Vertical line for calculated t-statistic (clamp to visible range)
    t_stat_clamped = max(x_min, min(t_stat, x_max))

    fig.add_vline(
        x=t_stat_clamped,
        line_dash='dash',
        line_color='#dc2626',
        line_width=3
    )

    # Add annotation for calculated t-statistic
    # Position annotation at clamped location but show actual t-statistic
    annotation_x = t_stat_clamped
    if t_stat < x_min:
        annotation_text = f"<b>Sample t = {t_stat:.2f}<br>(← off chart)</b>"
        ax_offset = 20
        annotation_x *= 0.84
    elif t_stat > x_max:
        annotation_text = f"<b>Sample t = {t_stat:.2f}<br>(off chart →)</b>"
        ax_offset = -20
        annotation_x *= 0.84
    else:
        annotation_text = f"<b>Sample t = {t_stat:.2f}</b>"
        ax_offset = 20 if t_stat > 0 else -20

    fig.add_annotation(
        x=annotation_x, y=max(y) * 0.9,
        text=annotation_text,
        showarrow=False,
        # arrowhead=2,
        # arrowcolor='#dc2626',
        # arrowwidth=2,
        ax=ax_offset,
        ay=-25  ,
        font=dict(color='#dc2626', size=12, family='monospace'),
        bgcolor='rgba(254, 242, 242, 0.9)',
        bordercolor='#dc2626',
        borderwidth=2
    )

    # Annotation for t=0 corresponds with μ₀
    fig.add_annotation(
        x=0, y=max(y) * 1.08,
        text=f"<b>t = 0 corresponds with μ₀={mu_0:.1f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#4b5563',
        font=dict(color='#1f2937', size=14),
        bgcolor='rgba(249, 250, 251, 0.9)',
        bordercolor='#6b7280',
        borderwidth=2
    )

    # Light theme layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937', size=12),
        xaxis=dict(
            title='<b>t-statistic</b>',
            gridcolor='#e5e7eb',
            zerolinecolor='#9ca3af',
            showgrid=True,
            zeroline=True,
            titlefont=dict(size=14, color='#111827'),
            tickfont=dict(size=11, color='#1f2937'),
            range=[x_min, x_max]
        ),
        yaxis=dict(
            title='<b>Probability Density</b>',
            gridcolor='#e5e7eb',
            zerolinecolor='#9ca3af',
            showgrid=True,
            titlefont=dict(size=14, color='#111827'),
            tickfont=dict(size=11, color='#1f2937')
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#d1d5db',
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=11, color='#1f2937')
        ),
        height=500,
        hovermode='x unified',
        margin=dict(t=80)
    )

    return fig

# Display visualization and results side by side
st.markdown("## t-Distribution Under H₀")

col1, col2 = st.columns([2, 1])

with col1:
    # Select Significance Level
    st.markdown("#### Select Significance Level (α)")

    alpha_col1, alpha_col2, alpha_col3 = st.columns(3)

    with alpha_col1:
        st.button("α = 0.01", use_container_width=True,
                  type="primary" if st.session_state.alpha == 0.01 else "secondary",
                  on_click=set_alpha, args=(0.01,))

    with alpha_col2:
        st.button("α = 0.05", use_container_width=True,
                  type="primary" if st.session_state.alpha == 0.05 else "secondary",
                  on_click=set_alpha, args=(0.05,))

    with alpha_col3:
        st.button("α = 0.10", use_container_width=True,
                  type="primary" if st.session_state.alpha == 0.10 else "secondary",
                  on_click=set_alpha, args=(0.10,))

    # st.markdown("")  # Add spacing

    # Display t-distribution plot
    fig = create_t_distribution_plot(df, st.session_state.alpha, t_stat, test_key)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### P-Value")
    st.markdown(f"""
    The p-value represents the probability of wrongfully rejecting H₀ when it is actually true:
    """)

    st.latex(rf"P(\text{{Wrongfully reject }} H_0 \mid H_0 \text{{ true}}) = {p_value:.6f}")

    st.markdown("")

    # Decision
    st.markdown("#### Decision")
    if decision == "Reject H₀":
        st.success(f"**{decision}** at α = {st.session_state.alpha}")
        st.markdown(f"Since p-value ({p_value:.6f}) < α ({st.session_state.alpha}), we reject the null hypothesis. "
                    f"The data provides sufficient evidence to support Hₐ.")
    else:
        st.error(f"**{decision}** at α = {st.session_state.alpha}")
        st.markdown(f"Since p-value ({p_value:.6f}) ≥ α ({st.session_state.alpha}), we fail to reject the null hypothesis. "
                    f"The data does not provide sufficient evidence to support Hₐ.")

# # Additional interpretation
# st.markdown("---")
# st.markdown("### Interpretation")
# with st.expander("Understanding the Results"):
#     st.markdown(f"""
#     **Your Test Setup:**
#     - Sample mean: {x_bar}
#     - Hypothesized population mean (μ₀): {mu_0}
#     - Sample standard deviation: {s}
#     - Sample size: {n}
#     - Test type: {test_key}
#     - Significance level: {alpha}

#     **What the t-statistic tells us:**
#     - The t-statistic ({t_stat:.4f}) measures how many standard errors the sample mean is from μ₀
#     - A t-statistic far from 0 suggests the sample mean is unlikely under H₀

#     **What the p-value tells us:**
#     - The p-value ({p_value:.6f}) is the probability of observing data as extreme as ours if H₀ were true
#     - A small p-value (< α) suggests the data is incompatible with H₀

#     **What the shaded region shows:**
#     - The shaded cyan region is the {(1-alpha)*100:.0f}% confidence region under H₀
#     - If the test statistic falls in this region, we fail to reject H₀
#     - If it falls outside (in the rejection region), we reject H₀
#     """)
