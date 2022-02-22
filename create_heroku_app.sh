#!/bin/bash
heroku create fastapi-ml-deployment --buildpack heroku/python
heroku buildpacks:add --index 1 heroku-community/apt

