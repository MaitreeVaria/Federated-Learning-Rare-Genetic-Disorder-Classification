# Federated Learning for Rare Genetic Disorder Classification

Welcome to the Federated Learning for Rare Genetic Disorder Classification project! This repository demonstrates the application of Federated Learning (FL) techniques to classify rare genetic disorders using Electronic Health Record (EHR) data while addressing two critical challenges: privacy and limited data availability.

## Project Overview

The primary aim of this project is to leverage Federated Learning to classify rare genetic disorders using EHR data from various hospitals, ensuring privacy preservation and overcoming data scarcity issues. Here are some key aspects of the project:

- **Data Source**: EHR data from various domains, including personal history, medical history, and family history, was used. You can find the dataset on Kaggle: [Link to Dataset](https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge).

## Project Structure

The project is organized as follows:

- **`clients/`**: This folder contains the client files where individual private EHR data is used to train a neural network. Each client represents a different hospital or data source.

- **`data/`**: All the individual data used by the clients is stored in this folder. Each client has its own data, ensuring data privacy.

- **`server.py`**: This file combines the weights sent by the clients and creates a global model. It acts as the central server for coordinating Federated Learning.

- **`model.h5`**: This file contains the trained classification model, which can be used for rare genetic disorder classification.

## Getting Started

To replicate or use this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/MaitreeVaria/Federated-Learning-Rare-Genetic-Disorder-Classification.git
   ```

2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the EHR data from the provided Kaggle dataset or your own data sources.

4. Customize the client files in the `clients/` folder to suit your data and privacy requirements.

5. Run the Federated Learning process by executing `server.py`. This will coordinate the training process across the clients.

6. Once the training is complete, you can use the `model.h5` file to classify rare genetic disorders.

## Contributing

Contributions to this project are welcome! If you have any improvements, bug fixes, or new features to propose, please open an issue or submit a pull request. Make sure to follow the project's coding standards and guidelines.

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for details. The Apache License 2.0 is an open-source license that allows you to use, modify, and distribute the code, subject to certain conditions and limitations specified in the license.

Thank you for your interest in the Federated Learning for Rare Genetic Disorder Classification project. If you have any questions or need assistance, please don't hesitate to reach out.

Happy Federated Learning!
