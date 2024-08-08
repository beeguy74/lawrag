# French Labor Code Search Engine

This project is a search engine for the French Labor Code. It provides two entry points: `load_data.py` for loading a CSV table as the data source, and `main.py` for querying the system.

## Getting Started

To get started with this project, follow the steps below:
To get started with this project, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Rename the `.env-example` file to `.env` and provide the necessary environment variables.
4. Run the `load_data.py` script to load the CSV table as the data source.
5. Execute the `main.py` script to start querying the system.

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the `load_data.py` script to load the CSV table as the data source.
4. Execute the `main.py` script to start querying the system.

## Usage

### Loading Data

To load the CSV table as the data source, run the following command and provide path to file.csv:

```bash
python load_data.py data/extrait_travail.csv
```

This script will parse the CSV file and populate the search engine with the relevant data.

### Querying the System

To query the search engine, use the `main.py` script. 
This script provides a command-line interface for searching the French Labor Code. Simply enter your search query, and the system will return the relevant results.
Run the following command and provide path to vector_storage from load_data.py script:

```bash
python main.py path/to/storage
```

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
