# InPar

## Overview

InPar is a robust system for parsing invoices using a combination of LayoutLM and YOLO models. The system is built with a Next.js frontend, a Django backend, and an invoice extractor that processes documents by subscribing to a Redis queue.

## Stack Used
- **Next.js** (Frontend)
- **Django** (Backend)
- **LayoutLM + YOLO** (Invoice Parsing Models)
- **Redis** (Queue Management)

## Components

### 1. django_backend
The backend component is responsible for handling document uploads, publishing request IDs to the Redis queue, and updating the database with the parsed data.

#### Features:
- Uploads documents to a local folder
- Publishes request IDs to a Redis queue
- Updates database with parsed data

#### Setup:
1. Navigate to the `django_backend` folder:
    ```sh
    cd django_backend
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the Django server:
    ```sh
    python manage.py runserver
    ```

For detailed documentation, refer to the [README.md](django_backend/README.md) file in the `django_backend` folder.

### 2. extractor
The extractor subscribes to the Redis queue and processes invoices using models built from LayoutLM and YOLO.

#### Features:
- Subscribes to Redis queue
- Parses invoices using LayoutLM and YOLO models

#### Setup:
1. Navigate to the `extractor` folder:
    ```sh
    cd extractor
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the extractor server:
    ```sh
    python extractor_server.py
    ```

For detailed documentation, refer to the [README.md](extractor/README.md) file in the `extractor` folder.

### 3. frontend
The frontend is built with Next.js and provides a user interface for interacting with the invoice parsing system.

#### Setup:
1. Navigate to the `frontend` folder:
    ```sh
    cd frontend
    ```
2. Install dependencies:
    ```sh
    npm install
    ```
3. Run the development server:
    ```sh
    npm run dev
    ```
4. Open your browser and go to:
    ```sh
    http://localhost:3000
    ```

## Getting Started
Follow the setup instructions for each component to get the system up and running. Ensure that the Django backend, extractor, and frontend are all running simultaneously for the full functionality of the system.

## Contributing
We welcome contributions to improve the Tally AI Document AI system. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```sh
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```sh
    git push origin feature-name
    ```
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any questions or support, please reach out to [psi.teja@gmail.com](mailto:psi.teja@gmail.com).
