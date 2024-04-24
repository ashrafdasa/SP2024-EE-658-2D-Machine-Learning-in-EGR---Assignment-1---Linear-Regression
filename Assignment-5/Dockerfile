FROM python:3
WORKDIR /app
COPY . .
RUN pip install streamlit

RUN pip install pypickle
RUN pip install streamlit-drawable-canvas==0.3.0
RUN pip install pandas
RUN pip install pillow
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install streamlit-vertical-slider
RUN pip install streamlit-toggle
RUN pip install jsons
RUN pip install pytest-warnings
 
CMD ["streamlit", "run", "Assignment-5-predict_page.py"]
