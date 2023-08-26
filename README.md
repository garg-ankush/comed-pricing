# comed-pricing

This project makse prediction requests on the Comed pricing at regular intervals. Based on the predictions, it sends out a message to subscribers if they should turn off their AC or heating.

# Components
- Makefile
- Requirements.txt
- test_library.py
- python_library
- Dockerfile
- Command line tool
- Microservice

# How to get set up
1. Create a virtual environment: virtualenv ~/.venv
2. Edit bashrc and source virtual env from it.
3. Run make all
4. Add .env file to the root. Follow the example in .env-example and fill the details in.
