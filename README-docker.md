# DARF System with Prometheus and Grafana

This setup provides a containerized version of the DARF System integrated with Prometheus for metrics collection and Grafana for visualization.

## Architecture

The system consists of three main components:

1. **DARF Dashboard** - A Flask-based web application that provides a UI for the DARF system with real metrics
2. **Prometheus** - Time-series database for storing metrics collected from the DARF system
3. **Grafana** - Visualization platform for creating dashboards based on Prometheus data

## Running with Docker Compose

### Prerequisites

- Docker and Docker Compose installed

### Starting the System

```bash
# Build and start all containers
docker-compose up -d

# Check if all containers are running
docker-compose ps
```

### Accessing the System

Once the system is running, you can access:

- **DARF Dashboard**: http://localhost:5000
- **Prometheus UI**: http://localhost:9090 
- **Grafana**: http://localhost:3000
  - Default credentials: admin/admin

### Stopping the System

```bash
# Stop all containers
docker-compose down

# Remove volumes (optional, will delete persistent data)
docker-compose down -v
```

## Manual Setup (without Docker)

If you prefer to run the system without Docker:

1. Install dependencies:
   ```
   pip install -r requirements_consolidated.txt
   ```

2. Install Prometheus and Grafana according to your operating system's instructions

3. Start the DARF dashboard:
   ```
   python darf_consolidated_dashboard.py
   ```

4. Update `prometheus/prometheus.yml` to use `localhost` instead of container names
   
5. Start Prometheus with the config file

6. Start Grafana and configure it to use your Prometheus instance

## Customizing the System

### Adding Custom Metrics

1. Edit the `metrics/darf_metrics.json` file to add custom metrics
2. Restart the DARF dashboard container:
   ```
   docker-compose restart darf-dashboard
   ```

### Creating Custom Dashboards

1. Log in to Grafana (http://localhost:3000)
2. Create a new dashboard using the Prometheus data source
3. Save your dashboard to the `grafana/dashboards` directory if you want to keep it in version control

## Troubleshooting

### Container not starting

Check the container logs:
```
docker-compose logs darf-dashboard
docker-compose logs prometheus
docker-compose logs grafana
```

### Metrics not showing up in Grafana

1. Check if metrics are being exposed by the DARF dashboard:
   ```
   curl http://localhost:5000/metrics
   ```

2. Check if Prometheus is scraping the metrics:
   - Open Prometheus UI (http://localhost:9090)
   - Go to Status > Targets to see if the scrape targets are up

3. Verify Prometheus is configured as a data source in Grafana:
   - Open Grafana (http://localhost:3000)
   - Go to Configuration > Data Sources to check if Prometheus is properly configured
