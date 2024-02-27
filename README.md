# Deployment of Decision Tree Model trained on Bike Rental Dataset
This is a project for continuous Docker deployment of decision tree model trained on rental bikes dataset.

## Main components and features of the code

* Using a data/ML pipeline to perform feature transformation/engineering, dropping correlated features, feature standardization and training on Decision Tree Regressor
* Performing automated testing using the config in tox.ini
* If the tests pass, the model will be published to Gemfury (hosted repository)
* Continuous Docker deployment to Railway app using CI/CD in CircleCI platform


## Data/ML pipleline steps

1. YeoJohnson transformation on numerical variables
2. one hot encoding of categorical variables
3. drop correlated features
4. standardization of all features
5. Training on decision tree regressor
