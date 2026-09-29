 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "student_performance.csv"
MODEL = BASE / "models" / "student_model.pkl"

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
:root {
    --bg: #f5f7fb;
    --card: #ffffff;
    --text: #172033;
    --muted: #667085;
    --primary: #4f46e5;
    --border: #e7eaf0;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1450px; }
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #202938;
}
[data-testid="stSidebar"] * { color: #f9fafb !important; }
.hero {
    background: linear-gradient(135deg,#111827 0%,#312e81 55%,#4f46e5 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 12px 30px rgba(17,24,39,.15);
}
.hero h1 { margin:0; font-size: 34px; }
.hero p { margin:8px 0 0; color:#e0e7ff; font-size:15px; }
.kpi {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(16,24,40,.05);
}
.kpi-label { color:#667085; font-size:13px; margin-bottom:6px; }
.kpi-value { color:#111827; font-size:28px; font-weight:750; }
.section {
    font-size: 20px; font-weight: 700; margin: 22px 0 12px;
}
.pred-pass {
    padding: 25px; border-radius: 18px; text-align:center;
    background:#ecfdf3; border:1px solid #abefc6; color:#067647;
}
.pred-fail {
    padding: 25px; border-radius: 18px; text-align:center;
    background:#fef3f2; border:1px solid #fecdca; color:#b42318;
}
.small-note { color:#667085; font-size:13px; }
.input-label {
    color:#172033 !important;
    font-size:14px !important;
    font-weight:700 !important;
    margin: 4px 0 5px 0 !important;
    display:block !important;
}
div[data-testid="stMetric"] {
    background:#fff; border:1px solid #e7eaf0; padding:14px;
    border-radius:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load ----------
if not MODEL.exists():
    st.error("Model file not found. Run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL)
model = bundle["model"]
metrics = bundle["metrics"]
df = pd.read_csv(DATA)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 EduPredict")
    st.caption("Student Performance Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Prediction", "Student Analytics", "Dataset", "Model Performance", "Insights", "About Project"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("ML-powered academic risk prediction")
    st.caption("Random Forest Classifier")

# ---------- Helpers ----------
def kpi(label, value):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Analytics</h1>
  <p>Predict academic outcomes, explore student trends, and identify performance risk using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    total = len(df)
    passed = (df.Result == "Pass").sum()
    failed = total - passed
    pass_rate = passed / total * 100

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Students", f"{total:,}")
    with c2: kpi("Pass Rate", f"{pass_rate:.1f}%")
    with c3: kpi("Passed", f"{passed:,}")
    with c4: kpi("Failed", f"{failed:,}")

    st.markdown('<div class="section">Performance Overview</div>', unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        counts = df["Result"].value_counts().rename_axis("Result").reset_index(name="Students")
        fig = px.pie(counts, names="Result", values="Students", hole=.58,
                     title="Pass vs Fail Distribution")
        fig.update_layout(margin=dict(t=55,b=10,l=10,r=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        avg = df[["Study_Hours","Attendance","Internal_Marks","Assignment_Marks","Previous_Performance"]].mean()
        chart = pd.DataFrame({"Factor": avg.index, "Average": avg.values})
        fig = px.bar(
    chart,
    x="Average",
    y="Factor",
    orientation="h",
    title="Average Student Profile",
    text_auto=".1f",
    color_discrete_sequence=["#0B1F3A"]
)
        fig.update_layout(margin=dict(t=55,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Key Relationships</div>', unsafe_allow_html=True)
    fig = px.scatter(df, x="Attendance", y="Internal_Marks", color="Result",
                     size="Study_Hours", hover_data=["Student_ID"],
                     title="Attendance vs Internal Marks")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Prediction ----------
elif page == "Prediction":
    st.markdown('<div class="section">🔮 Individual Student Prediction</div>', unsafe_allow_html=True)
    st.write("Enter the student's academic details. The model estimates the probability of passing.")
    st.caption("Adjust each factor using the sliders below. All five inputs are used by the machine-learning model.")
    left,right = st.columns([1.15, .85])

    with left:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="input-label">1. Study Hours / Day</div>', unsafe_allow_html=True)
            study = st.slider("Study Hours / Day", 0.0, 12.0, 5.0, 0.5, label_visibility="collapsed")

            st.markdown('<div class="input-label">2. Attendance (%)</div>', unsafe_allow_html=True)
            attendance = st.slider("Attendance (%)", 0, 100, 75, 1, label_visibility="collapsed")

            st.markdown('<div class="input-label">3. Internal Marks</div>', unsafe_allow_html=True)
            internal = st.slider("Internal Marks", 0, 100, 65, 1, label_visibility="collapsed")

        with c2:
            st.markdown('<div class="input-label">4. Assignment Marks</div>', unsafe_allow_html=True)
            assignment = st.slider("Assignment Marks", 0, 100, 70, 1, label_visibility="collapsed")

            st.markdown('<div class="input-label">5. Previous Performance</div>', unsafe_allow_html=True)
            previous = st.slider("Previous Performance", 0, 100, 65, 1, label_visibility="collapsed")

        if st.button("🚀 Predict Result", type="primary", use_container_width=True):
            X_new = pd.DataFrame([{
                "Study_Hours": study,
                "Attendance": attendance,
                "Internal_Marks": internal,
                "Assignment_Marks": assignment,
                "Previous_Performance": previous
            }])
            pred = int(model.predict(X_new)[0])
            probs = model.predict_proba(X_new)[0]
            pass_prob = probs[1] * 100
            fail_prob = probs[0] * 100

            st.session_state["prediction"] = (pred, pass_prob, fail_prob)

    with right:
        st.markdown("### Prediction Result")
        if "prediction" in st.session_state:
            pred, pass_prob, fail_prob = st.session_state["prediction"]
            if pred == 1:
                st.markdown(f"""
                <div class="pred-pass">
                  <div style="font-size:44px">✓</div>
                  <h2 style="margin:4px 0">PASS</h2>
                  <b>Pass probability: {pass_prob:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-fail">
                  <div style="font-size:44px">!</div>
                  <h2 style="margin:4px 0">FAIL RISK</h2>
                  <b>Fail probability: {fail_prob:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)
            st.progress(int(pass_prob))
            st.caption(f"Pass: {pass_prob:.1f}%  •  Fail: {fail_prob:.1f}%")
        else:
            st.info("Fill in the student details and click Predict Result.")

# ---------- Analytics ----------
elif page == "Student Analytics":
    st.markdown('<div class="section">📊 Student Analytics</div>', unsafe_allow_html=True)
    factors = ["Study_Hours","Attendance","Internal_Marks","Assignment_Marks","Previous_Performance"]
    selected = st.selectbox("Select performance factor", factors, format_func=lambda x: x.replace("_"," "))

    a,b = st.columns(2)
    with a:
        fig = px.histogram(df, x=selected, color="Result", nbins=20,
                           marginal="box", title=f"{selected.replace('_',' ')} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        means = df.groupby("Result")[selected].mean().reset_index()
        fig = px.bar(means, x="Result", y=selected, text_auto=".1f",
                     title=f"Average {selected.replace('_',' ')} by Result")
        st.plotly_chart(fig, use_container_width=True)

    corr = df[factors].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Dataset ----------
elif page == "Dataset":
    st.markdown('<div class="section">🗂 Student Dataset</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: result_filter = st.multiselect("Result", ["Pass","Fail"], default=["Pass","Fail"])
    with c2: min_att = st.slider("Minimum Attendance", 0, 100, 0)
    with c3: search_id = st.text_input("Search Student ID", placeholder="e.g. 101")

    view = df[df["Result"].isin(result_filter) & (df["Attendance"] >= min_att)].copy()
    if search_id.strip():
        try:
            sid = int(search_id)
            view = view[view.Student_ID == sid]
        except:
            view = view.iloc[0:0]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download Filtered CSV", view.to_csv(index=False).encode(), "filtered_students.csv", "text/csv")

# ---------- Model ----------
elif page == "Model Performance":
    st.markdown('<div class="section">🤖 Machine Learning Model</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    with c2: kpi("Precision", f"{metrics['precision']*100:.1f}%")
    with c3: kpi("Recall", f"{metrics['recall']*100:.1f}%")
    with c4: kpi("F1 Score", f"{metrics['f1']*100:.1f}%")

    cm = np.array(metrics["confusion_matrix"])
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted Fail","Predicted Pass"],
        y=["Actual Fail","Actual Pass"], text=cm, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Features")
    st.write("The model uses study hours, attendance, internal marks, assignment marks, and previous performance.")
    st.info("Random Forest is used because it can model non-linear relationships between academic factors and outcomes.")

# ---------- Insights ----------
elif page == "Insights":
    st.markdown('<div class="section">💡 Academic Insights</div>', unsafe_allow_html=True)
    passed = df[df.Result == "Pass"]
    failed = df[df.Result == "Fail"]
    factors = ["Study_Hours","Attendance","Internal_Marks","Assignment_Marks","Previous_Performance"]

    insight_rows = []
    for f in factors:
        diff = passed[f].mean() - failed[f].mean()
        insight_rows.append([f.replace("_"," "), passed[f].mean(), failed[f].mean(), diff])
    ins = pd.DataFrame(insight_rows, columns=["Factor","Pass Avg","Fail Avg","Difference"])
    st.dataframe(ins.style.format({"Pass Avg":"{:.1f}","Fail Avg":"{:.1f}","Difference":"{:.1f}"}),
                 use_container_width=True, hide_index=True)

    st.info("Use these comparisons to identify which academic areas may need attention. The model prediction is an estimate, not a guarantee of a student's final result.")

# ---------- About ----------
elif page == "About Project":
    st.markdown('<div class="section">📘 About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    **Problem Statement**

    A college wants to predict whether a student will **Pass or Fail** based on academic and engagement factors.

    **Objectives**
    - Predict student outcome using machine learning.
    - Analyze academic performance patterns.
    - Provide an easy-to-use prediction dashboard.
    - Help identify students who may need additional academic support.

    **Input Features**
    - Study Hours
    - Attendance
    - Internal Marks
    - Assignment Marks
    - Previous Performance

    **Technology Stack**
    - Python
    - Pandas & NumPy
    - Scikit-learn
    - Random Forest
    - Streamlit
    - Plotly
    - Joblib

    **Workflow**

    `Student Data → Data Preprocessing → Train/Test Split → Random Forest → Prediction → Dashboard`

    **Important:** The included dataset is synthetic/demo data created for the college project. It should not be presented as real institutional student records.
    """)

st.markdown("---")
st.caption("EduPredict • Student Performance Prediction System • Academic Project")
