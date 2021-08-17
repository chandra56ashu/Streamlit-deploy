import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def prediction(message):
    input= ['message']
    pred=model.predict(input)
    return str(prediction)

def main():
    st.title("Sentiments")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiments prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    message = st.text_input("message","Type Here")
    safe_html="""  
      <div style="background-color:#16A085;padding:10px >
       <h2 style="color:green;text-align:center;"> Positive</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#641E16;padding:10px >
       <h2 style="color:red ;text-align:center;"> Negative</h2>
       </div>
    """
    neutral_html="""  
      <div style="background-color:#F1C40F;padding:10px >
       <h2 style="color:yellow ;text-align:center;"> Neutral</h2>
       </div>
    """

    if st.button("Predict"):
        output=prediction(message)
        st.success('The message is  {}'.format(output))

        if output == 1:
            st.markdown(danger_html,unsafe_allow_html=True)
        elif output == 0:
            st.markdown(neutral_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()