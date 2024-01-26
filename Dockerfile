FROM python:3.11

# Create the user that will run the app
RUN adduser --disabled-password --gecus '' dt-ml-api-user

WORKDIR /opt/decision_tree_api

ARG PIP_EXTRA_INDEX_URL

# install requirements, copying from the local dir to the container, including from Gemfury
ADD ./decision_tree_api /opt/decision_tree_api
RUN pip install --upgrade pip
RUN pip install -r /opt/house-prices-api/requirements.txt

RUN chmod +x /opt/decision_tree_api/run.sh
RUN chmod -R dt-ml-api-user:dt-ml-api-user ./

USER dt-ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
