

#create a streamlit app to show how EMG can be analysed


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt


def high_pass_filter(signal, cutoff, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


st.title('Analyze electromyography (EMG) data')

"This program will walk you through the steps of EMG of gait data from 4 different leg muscles"
# Display image at the top of the app
st.image("images/muscles_of_leg.png", use_container_width=True)

# Set random seed for reproducibility
np.random.seed(42)

# Time axis: 5 gait cycles (each cycle normalized to 0–1)
gait_cycles = 5
samples_per_cycle = 1000
t_single = np.linspace(0, 1, samples_per_cycle)
t = np.tile(t_single, gait_cycles)
total_time = gait_cycles
time_vector = np.linspace(0, total_time, gait_cycles * samples_per_cycle)

# Function to create asymmetric bursts
def create_burst(center, width, scale=1.0):
    burst = np.exp(-((t_single - center) ** 2) / (2 * width ** 2))
    return burst * scale

# Define realistic muscle activation profiles (per cycle)
def generate_muscle_profile(patterns, noise_level=0.05):
    signal = np.zeros_like(t)
    for i in range(gait_cycles):
        cycle_base = i * samples_per_cycle
        cycle_signal = np.zeros(samples_per_cycle)
        for (center, width, scale) in patterns:
            cycle_signal += create_burst(center, width, scale)
        # Add variable noise
        noise = noise_level * np.random.randn(samples_per_cycle)
        signal[cycle_base:cycle_base + samples_per_cycle] = cycle_signal + noise
    return signal

# EMG activation patterns: (center, width, amplitude)
ta_pattern = [(0.05, 0.03, 0.6), (0.8, 0.05, 0.4)]    # TA: early stance & swing
gas_pattern = [(0.45, 0.07, 0.8)]                    # GAS: push-off
bf_pattern = [(0.9, 0.04, 0.6), (0.1, 0.03, 0.3)]     # BF: late swing, early stance
rf_pattern = [(0.15, 0.05, 0.6), (0.85, 0.05, 0.4)]   # RF: early stance, late swing

# Generate signals
ta_signal = generate_muscle_profile(ta_pattern)
gas_signal = generate_muscle_profile(gas_pattern)
bf_signal = generate_muscle_profile(bf_pattern)
rf_signal = generate_muscle_profile(rf_pattern)



# Streamlit interface
st.title("Gait Cycle EMG Data Analaysis")
muscle_options = ["Tibialis Anterior (TA)", "Gastrocnemius (GM)", "Biceps Femoris (BF)", "Rectus Femoris (RF)"]
selected_muscle = st.selectbox("Select Muscle to Analyze", muscle_options)

# Map muscle to signal
muscle_map = {
    "Tibialis Anterior (TA)": ta_signal,
    "Gastrocnemius (GM)": gas_signal,
    "Biceps Femoris (BF)": bf_signal,
    "Rectus Femoris (RF)": rf_signal
}
muscle_signal = muscle_map[selected_muscle]

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_vector, y=muscle_signal, mode='lines', name=selected_muscle))

for i in range(gait_cycles):
    fig.add_vline(x=i, line=dict(color="black", dash="dot"),
                  annotation_text=f"Cycle {i+1}", annotation_position="top right")

fig.update_layout(
    title=f"EMG for {selected_muscle}",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (µV)",
    showlegend=True
)

st.plotly_chart(fig)

st.write("""
The first step in signal analysis is to clean up the signal to remove noise and focus on the relevant motion data
""")


# ------------------------------
# Step 1: High-Pass Filtering
# ------------------------------
st.markdown("## Step 1: High-Pass Filtering")

st.write("""
A high-pass filter removes low-frequency movement artifacts and DC drift.  
Use the sliders below to explore how different cutoff frequencies affect the signal.
""")

apply_filter = st.checkbox("Apply High-Pass Filter")

if apply_filter:
    cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=1, max_value=30, value=20, step=1)
    filter_order = st.slider("Filter Order", min_value=1, max_value=8, value=4)

    # Define high-pass filter function
    def high_pass_filter(signal, cutoff, fs=1000, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal)

    # Apply high-pass filter
    filtered_signal = high_pass_filter(muscle_signal, cutoff=cutoff_freq, order=filter_order)

    # Plot filtered signal
    fig_filt = go.Figure()
    fig_filt.add_trace(go.Scatter(x=time_vector, y=filtered_signal, mode='lines', name=f"{selected_muscle} (Filtered)"))

    for i in range(gait_cycles):
        fig_filt.add_vline(x=i, line=dict(color="black", dash="dot"),
                        annotation_text=f"Cycle {i+1}", annotation_position="top right")

    fig_filt.update_layout(
        title=f"High-Pass Filtered EMG for {selected_muscle}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=True
    )

    st.plotly_chart(fig_filt)
    st.write(f"High-pass filter applied with cutoff = {cutoff_freq} Hz and order = {filter_order}.")
else:
    st.info("High-pass filter not applied. You can enable it to explore filtering effects.")



# ------------------------------
# Step 2: Full wave rectification
# ------------------------------

st.markdown("## Step 2: Full Wave Rectification")

st.write("""
EMG signals from bipolar electrodes contain both positive and negative voltages due to muscle fiber depolarization. If we averaged this raw signal directly, the positive and negative phases would cancel each other out. To avoid this and accurately reflect overall muscle activity, we apply full-wave rectification, which converts all values to positive by taking the absolute value of the signal
""")

"Click the box below to rectify the signal"

apply_rectification = st.checkbox("Apply Full-Wave Rectification")

if apply_rectification:
    muscle_signal_rect = np.abs(filtered_signal) if apply_rectification else muscle_signal
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vector, y=muscle_signal_rect, mode='lines', name=selected_muscle))

    for i in range(gait_cycles):
        fig.add_vline(x=i, line=dict(color="black", dash="dot"),
                      annotation_text=f"Cycle {i+1}", annotation_position="top right")

    fig.update_layout(
        title=f"EMG for {selected_muscle}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=True
    )

    st.plotly_chart(fig)

    st.write(f"{'Full-Wave Rectification applied.' if apply_rectification else 'Raw EMG signal shown.'}")
else:
    st.info("Full-wave rectification not applied. You can enable it by clicking the checkbox.")

# ------------------------------
# Step 3: Low-Pass Filtering (Envelope Detection)
# ------------------------------
st.markdown("## Step 3: Low-Pass Filtering (Envelope Detection)")

st.write("""
Low-pass filtering smooths the rectified EMG signal to reveal its **envelope**, which reflects the overall trend of muscle activation over time.
This is commonly used to analyze muscle timing and intensity in gait.
""")

apply_lowpass = st.checkbox("Apply Low-Pass Filter (Envelope Detection)")

if apply_lowpass:
    lp_cutoff = st.slider("Low-Pass Cutoff Frequency (Hz)", min_value=1, max_value=300, value=6, step=1)
    lp_order = st.slider("Low-Pass Filter Order", min_value=1, max_value=8, value=4)

    # Define low-pass filter function
    def low_pass_filter(signal, cutoff, fs=1000, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    # Choose input: filtered signal from Step 2 or signal_to_plot from Step 1
    input_signal = filtered_signal if apply_filter else signal_to_plot
    envelope_signal = low_pass_filter(input_signal, cutoff=lp_cutoff, order=lp_order)

    # Plot envelope
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=time_vector, y=envelope_signal, mode='lines', name="EMG Envelope"))

    for i in range(gait_cycles):
        fig_env.add_vline(x=i, line=dict(color="black", dash="dot"),
                          annotation_text=f"Cycle {i+1}", annotation_position="top right")

    fig_env.update_layout(
        title=f"EMG Envelope for {selected_muscle} (Low-Pass Filtered)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=True
    )

    st.plotly_chart(fig_env)
    st.write(f"Low-pass filter applied with cutoff = {lp_cutoff} Hz and order = {lp_order}.")
else:
    st.info("Low-pass filter not applied. You can enable it to view the EMG envelope.")



# ------------------------------
# Step 4: Gait Cycle Averaging
# ------------------------------
st.markdown("## Step 4: Gait Cycle Averaging")

st.write("""
Gait cycle averaging (also called phase or ensemble averaging) helps identify consistent activation patterns by reducing noise.
This is done by dividing the signal into individual gait cycles and averaging them.
""")
st.write("""
0% corresponds to the start of the gait cycle, usually defined as Initial Contact (heel strike) of one foot.
100% corresponds to the moment just before that same foot contacts the ground again — completing one full step.
""")
apply_avg = st.checkbox("Apply Gait Cycle Averaging")

if apply_avg:
    # Choose input signal from previous steps
    base_signal = envelope_signal if apply_lowpass else (
        filtered_signal if apply_filter else signal_to_plot
    )

    # Reshape into [cycles x samples] for averaging
    cycle_samples = samples_per_cycle
    segments = base_signal.reshape((gait_cycles, cycle_samples))
    mean_signal = segments.mean(axis=0)
    time_normalized = np.linspace(0, 100, cycle_samples)

    # Plot individual cycles and the average
    fig_avg = go.Figure()
    show_individual = st.checkbox("Show Individual Gait Cycles", value=True)

    if show_individual:
        for i in range(gait_cycles):
            fig_avg.add_trace(go.Scatter(
                x=time_normalized, y=segments[i],
                mode='lines', name=f"Cycle {i+1}",
                line=dict(width=1), opacity=0.4
            ))

    # Add average signal
    fig_avg.add_trace(go.Scatter(
        x=time_normalized, y=mean_signal,
        mode='lines', name="Average",
        line=dict(color='black', width=3)
    ))

    fig_avg.update_layout(
        title=f"Gait Cycle Averaged EMG for {selected_muscle}",
        xaxis_title="Gait Cycle (%)",
        yaxis_title="Amplitude (µV)",
        showlegend=True
    )

    st.plotly_chart(fig_avg)
    st.write("Each cycle has been time-normalized to 0–100% and averaged across all steps.")
else:
    st.info("Enable cycle averaging to see the average EMG waveform across gait cycles.")



# ------------------------------
# Step 5: Identify Gait Events (Multiple Choice)
# ------------------------------
# ------------------------------
# Step 5: Identify Gait Events (Multiple Choice with Highlight)
# ------------------------------
st.markdown("## Step 5: Identify Key Gait Events")

st.write("""
Use the average EMG waveform to help identify where key gait events occur.  
Select the **correct percentage range** for each event below, then see your choice highlighted on the graph.
""")

apply_event_quiz = st.checkbox("Start Gait Event Identification")

if apply_event_quiz:
    # Use average signal from Step 4
    quiz_signal = mean_signal if apply_avg else None

    if quiz_signal is None:
        st.warning("Please complete Step 4: Gait Cycle Averaging first.")
    else:
        # Plot the average EMG (initial static plot for reference)
        quiz_fig = go.Figure()
        quiz_fig.add_trace(go.Scatter(
            x=time_normalized, y=quiz_signal, mode='lines', name="Average EMG"
        ))

        quiz_fig.update_layout(
            title="Gait Cycle: 0–100% (Average EMG)",
            xaxis_title="Gait Cycle (%)",
            yaxis_title="Amplitude (µV)"
        )

        st.plotly_chart(quiz_fig)

        # Correct answer ranges
        correct_answers = {
            "Initial Contact": "0–10%",
            "Toe-Off": "60–70%",
            "Swing Phase": "60–100%",
            "Stance Phase": "0–60%"
        }

        # Options for dropdown (with default choice)
        options = ["0–10%", "10–30%", "30–60%", "60–70%", "60–100%", "0–60%", "70–100%"]
        all_options = ["Choose an option"] + options

        # Map options to numeric ranges for highlighting
        range_map = {
            "0–10%": (0, 10),
            "10–30%": (10, 30),
            "30–60%": (30, 60),
            "60–70%": (60, 70),
            "60–100%": (60, 100),
            "0–60%": (0, 60),
            "70–100%": (70, 100)
        }

        # Loop through events for user to select answers
        for event, correct_range in correct_answers.items():
            st.markdown(f"**{event}**")
            user_choice = st.selectbox(
                f"Select the range where you think {event} occurs:",
                all_options,
                key=event
            )

            if user_choice != "Choose an option":
                if user_choice == correct_range:
                    st.success(f"✅ Correct! {event} typically occurs at {correct_range}.")
                else:
                    st.error(f"❌ Not quite. {event}. Please try again.")

                # Highlight the selected range on a new plot
                start_perc, end_perc = range_map[user_choice]

                fig_highlight = go.Figure()
                fig_highlight.add_trace(go.Scatter(
                    x=time_normalized, y=quiz_signal, mode='lines', name="Average EMG"
                ))

                fig_highlight.add_shape(
                    type="rect",
                    x0=start_perc, x1=end_perc,
                    y0=min(quiz_signal), y1=max(quiz_signal),
                    fillcolor="LightSalmon",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                )

                fig_highlight.update_layout(
                    title=f"{event}: Selected Range Highlighted",
                    xaxis_title="Gait Cycle (%)",
                    yaxis_title="Amplitude (µV)"
                )

                st.plotly_chart(fig_highlight)







# ------------------------------
# Step 6: Muscle Function Quiz
# ------------------------------
st.markdown("## Step 6: Muscle Function Quiz")

st.write(f"""
Based on your analysis of the EMG for **{selected_muscle}**, select its **primary function during gait**.
""")

# Define primary functions for muscles
muscle_functions = {
    "Tibialis Anterior (TA)": [
        "Dorsiflexion of the ankle during swing phase",
        "Plantarflexion during push-off",
        "Knee flexion during stance",
        "Hip extension during swing"
    ],
    "Gastrocnemius (GAS)": [
        "Plantarflexion during push-off",
        "Dorsiflexion during swing phase",
        "Knee extension during stance",
        "Hip flexion during swing"
    ],
    "Biceps Femoris (BF)": [
        "Knee flexion during swing and hip extension during stance",
        "Ankle dorsiflexion during swing",
        "Plantarflexion during stance",
        "Hip flexion during swing"
    ],
    "Rectus Femoris (RF)": [
        "Knee extension during stance and hip flexion during swing",
        "Ankle dorsiflexion during swing",
        "Plantarflexion during push-off",
        "Knee flexion during swing"
    ]
}

# Correct answers for validation
correct_answers = {
    "Tibialis Anterior (TA)": "Dorsiflexion of the ankle during swing phase",
    "Gastrocnemius (GAS)": "Plantarflexion during push-off",
    "Biceps Femoris (BF)": "Knee flexion during swing and hip extension during stance",
    "Rectus Femoris (RF)": "Knee extension during stance and hip flexion during swing"
}

options = muscle_functions[selected_muscle]
user_answer = st.selectbox("Select the primary function:", ["Choose an option"] + options)

if user_answer != "Choose an option":
    if user_answer == correct_answers[selected_muscle]:
        st.success("✅ Correct! Well done.")
    else:
        st.error(f"❌ Not quite. Please try again.")





