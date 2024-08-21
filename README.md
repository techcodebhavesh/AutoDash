# AutoDash

AutoDash is an advanced tool designed to revolutionize data analysis by automating the creation of interactive dashboards from multiple data sources. Leveraging a fine-tuned Llama3 LLM specifically tailored for data analytics, AutoDash transforms complex data into actionable insights through real-time updates and intuitive natural language interactions. This tool prioritizes security, ensuring that all data processing stays within the company's servers, providing dynamic visualizations and empowering businesses to make informed decisions effortlessly.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Backend Repository - Python (AutoDash API)](#1-backend-repository---python-autodash-api)
  - [Frontend Repository - React (AutoDash Frontend)](#2-frontend-repository---react-autodash-frontend)
  - [Backend Repository - Java (AutoDash Java API)](#3-backend-repository---java-autodash-java-api)
- [Tech Stack](#tech-stack)
- [Usage](#usage)
- [Contribution Guidelines](#contribution-guidelines)
- [FAQs](#faqs)
- [Roadmap](#roadmap)

---

## Project Overview

AutoDash is a cutting-edge solution designed to simplify the data analysis process by automating the creation of interactive, real-time dashboards. With a secure in-house processing system, AutoDash ensures that your data remains within your company’s infrastructure, offering both efficiency and peace of mind. The tool's AI-driven insights and dynamic visualizations empower businesses to make informed decisions quickly and accurately.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **Node.js** and **NPM**
- **JDK 21**
- **Maven**
- **MySQL**
- **Nginx**

---

## Setup Instructions

### 1. Backend Repository - Python (AutoDash API)

#### Repository Link: [AutoDash API](https://github.com/techcodebhavesh/AutoDash)

**Steps:**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/techcodebhavesh/AutoDash.git
   cd AutoDash
   ```

2. **Create a Python Virtual Environment**:
   ```bash
   python3 -m venv pyenv
   source pyenv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file with your credentials and other required configurations.

5. **Run the Flask Server**:
   ```bash
   python run.py
   ```

6. **Nginx Setup**:
   - Ensure Nginx is installed and configured to proxy requests to the Flask server.

---

### 2. Frontend Repository - React (AutoDash Frontend)

#### Repository Link: [AutoDash Frontend](https://github.com/Vaishnavi4008/Autodash_frontend)

**Steps:**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vaishnavi4008/Autodash_frontend.git
   cd Autodash_frontend
   ```

2. **Environment Configuration**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file with your credentials or any required configuration changes.

3. **Install Dependencies**:
   ```bash
   npm install
   ```

4. **Run the Development Server**:
   ```bash
   npm run dev
   ```
   - This will start the Vite development server, and the frontend will be accessible at `http://localhost:5173`.

---

### 3. Backend Repository - Java (AutoDash Java API)

#### Repository Link: [AutoDash Java API](https://github.com/SpectacularVoyager/AutodashJava)

**Steps:**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SpectacularVoyager/AutodashJava.git
   cd AutodashJava
   ```

2. **Install JDK 21**:
   - Ensure that JDK 21 is installed on your system.

3. **Build the Project**:
   ```bash
   mvn clean install
   ```

4. **Run the Spring Boot Application**:
   ```bash
   mvn spring-boot:run
   ```

5. **Database Setup**:
   - SQL files are located in the `res/sql.sql` directory.
   - Configure MySQL settings in `src/main/resources/jdbc.properties`.

---

## Tech Stack

- **Backend**: Flask (Python), Spring Boot (Java)
- **Frontend**: React (JavaScript), Vite
- **Database**: MySQL
- **Other**: Nginx, Maven, JDK 21

---

## Usage

Once all services are running, you can access AutoDash through your browser at `http://localhost:5173`.

- **Data Integration**: Connect multiple data sources through the dashboard.
- **Natural Language Queries**: Use the fine-tuned Llama3 LLM for intuitive data queries.
- **Dynamic Visualizations**: Customize and interact with data visualizations in real time.

---

## Contribution Guidelines

We welcome contributions to AutoDash! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

For more details, refer to our `CONTRIBUTING.md` file.

---

## FAQs

**Q: What if I encounter a `ModuleNotFoundError`?**
A: Ensure all dependencies are installed as per the `requirements.txt` or `package.json` files.

**Q: How do I configure MySQL for the Java backend?**
A: Edit the `jdbc.properties` file in `src/main/resources` with your MySQL credentials.

**Q: How do I set up Nginx for the Python backend?**
A: Follow standard Nginx setup procedures, ensuring it proxies requests to the Flask server.

---

## Roadmap

- **Version 2.0**: Multi-tenant support and enhanced security features.
- **Version 3.0**: Expanded LLM capabilities for more complex data queries.
- **Future Plans**: Integration with more data sources and enhanced visualization options.
