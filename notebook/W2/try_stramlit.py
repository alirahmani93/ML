import numpy as np
import pandas as pd
import streamlit as st

_balloons = st.balloons()
st.title("Hello Steamlit!")
with st.divider() as div1:
    st.status('this is running status')
st.status('this is complete status', state='complete')
st.status('this is error status', state='error')

with st.form("my_form"):
    st.caption("This is test caption for my form")
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

checkbox_val2 = st.checkbox("Outside checkbox")
st.write("slider", slider_val, "checkbox", checkbox_val2)

st.audio('./notebook/W2/Electric-shutdown-sound-effect.mp3', format='mp3')
st.button("Close", on_click=print('close'))
st.code('print("This is Code and Data")')
st.chat_input('please enter your text')
st.chat_message(name='ai', avatar='assistant')

st.color_picker(label='Choose your color', value='#FFAAAA', key='user_color')

df_tips = pd.read_csv('./datasets/tips.csv')
st.dataframe(data=df_tips)

st.button('Download', on_click=st.snow)
with st.container() as ct:
    st.latex('LaTeX Documentation')
    series = np.random.randn(50, 3)
    st.data_editor(series)
    st.caption('data_editor')
    st.bar_chart(series)

date = st.date_input('Please enter')

with st.container() as map_chart:
    st.divider()
    st.header('Map Chart')
    df = pd.DataFrame({
        "col1": np.random.randn(1000) / 50 + 35.761539,
        "col2": np.random.randn(1000) / 50 + 51.395015,
        "col3": np.random.randn(1000) * 100,
        "col4": np.random.rand(1000, 4).tolist(),
    })
    st.map(df,
           latitude='col1',
           longitude='col2',
           size='col3',
           color='col4')

st.divider()
tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
st.divider()

st.toast(body='Toast', icon='😍')
st.slider('Slider is HERE')
st.spinner('SPINNER')
st.info('info: A lot of info is here')
st.progress(value=10, text="Loading ...")
st.success('successfully Streamlit')

run_count = 0
while run_count < 3:
    run_count += 1
    print(run_count)
    # st.experimental_rerun()

# st.cache_resource()
with st.echo():
    st.write("ECHO economics")

# st.help()

col10, col20, col30 = st.columns(3)
with col10:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")
with col20:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")
with col30:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")
st.divider()

col1, col2 = st.columns([3, 1])
data = np.random.randn(10, 1)
col1.subheader("A wide column with a chart")
col1.line_chart(data)
col2.subheader("A narrow column with the data")
col2.write(data)

st.write("---") # equals to st.divider()