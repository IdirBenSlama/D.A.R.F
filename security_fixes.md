# Docker Security Vulnerabilities Remediation Plan

## Summary of Scan Results

The vulnerability scan identified security issues in two Docker images:

1. **alpine:3.20**
2. **rancher/mirrored-grafana-grafana:11.5.2**

### High Severity Vulnerabilities

| CVE ID | Severity | Package | Version | Description |
|--------|----------|---------|---------|-------------|
| CVE-2025-26519 | 8.1 (High) | musl | 1.2.5-r0 | High severity vulnerability in Alpine's musl library |
| CVE-2025-22869 | 7.5 (High) | golang.org/x/crypto | 0.32.0 | High severity vulnerability in Go crypto library |
| CVE-2025-22868 | 7.5 (High) | golang.org/x/oauth2 | 0.25.0 | High severity vulnerability in Go OAuth2 library |

### Medium Severity Vulnerabilities

| CVE ID | Severity | Package | Version | Description |
|--------|----------|---------|---------|-------------|
| CVE-2024-12797 | 6.3 (Medium) | openssl | 3.3.2-r1 | Medium severity vulnerability in OpenSSL |
| CVE-2024-56323 | 5.8 (Medium) | github.com/openfga/openfga | 1.6.2 | Medium severity vulnerability |
| CVE-2025-25196 | 5.8 (Medium) | github.com/openfga/openfga | 1.6.2 | Medium severity vulnerability |
| CVE-2020-8911 | 5.6 (Medium) | github.com/aws/aws-sdk-go | 1.55.5 | Medium severity vulnerability in AWS SDK |
| CVE-2025-22870 | 4.4 (Medium) | golang.org/x/net | 0.34.0 | Medium severity vulnerability in Go net library |
| CVE-2020-8552 | 4.3 (Medium) | k8s.io/apiserver | 0.32.0 | Medium severity vulnerability in Kubernetes apiserver |
| CVE-2024-13176 | 4.1 (Medium) | openssl | 3.3.2-r1 | Medium severity vulnerability in OpenSSL |

## Impact Analysis

### Current Docker Configuration

Your current Docker setup uses:
- `prom/prometheus:latest` for Prometheus
- `grafana/grafana:latest` for Grafana
- Custom image based on `python:3.9-slim` for DARF dashboard

While your docker-compose.yml uses `grafana/grafana:latest`, the scan identified issues in `rancher/mirrored-grafana-grafana:11.5.2`, which might be:
1. A cached/previously used image on your system
2. A mirror of the Grafana image used in a Kubernetes environment (Rancher)

The Alpine image vulnerabilities suggest either:
1. It's a base image for one of your other containers
2. It's a standalone container used elsewhere in your infrastructure

## Remediation Plan

### 1. Update Alpine-based Images

For the `alpine:3.20` image with CVE-2025-26519 (musl 1.2.5-r0):

```dockerfile
# Update to latest Alpine with fixed musl version
FROM alpine:3.20.5
```

Alternatively, consider switching to a distroless or minimal Debian-based image if Alpine isn't specifically required.

### 2. Grafana Image Update

For the Grafana image with multiple vulnerabilities:

```yaml
# In docker-compose.yml
services:
  grafana:
    # Use official fixed version instead of latest
    image: grafana/grafana:11.5.3   # Use the latest patched version
    # ... rest of your configuration
```

If you're using Rancher in a Kubernetes environment:
- Update the Rancher Helm charts or manifests to use the latest patched Grafana image
- Consider switching from the mirrored image to the official Grafana image if possible

### 3. Fix Golang Dependencies

For the Go-based applications with vulnerable dependencies:

1. Upgrade golang.org/x/crypto to version >0.32.0
2. Upgrade golang.org/x/oauth2 to version >0.25.0
3. Upgrade golang.org/x/net to version >0.34.0
4. Upgrade github.com/aws/aws-sdk-go to version >1.55.5
5. Upgrade github.com/openfga/openfga to a patched version

If you're not directly maintaining these Go applications but they're part of the Grafana image, updating to a newer Grafana version should include the fixes.

### 4. OpenSSL Vulnerabilities

For the OpenSSL vulnerabilities (CVE-2024-12797, CVE-2024-13176):

```dockerfile
# For Alpine-based images
RUN apk update && apk upgrade openssl
```

```dockerfile
# For Debian-based images
RUN apt-get update && apt-get upgrade -y openssl
```

## Implementation Steps

1. **Test Environment Updates**:
   - Create a testing environment to validate the image updates
   - Verify all functionality works with updated images

2. **Update Development Environment**:
   - Update docker-compose.yml with fixed image versions
   - Update Dockerfile.darf if needed
   - Test application functionality

3. **Production Deployment**:
   - Schedule maintenance window for production update
   - Update production Docker images
   - Monitor for any issues post-update

## Docker Security Best Practices

1. **Use Specific Image Tags**: Avoid using `:latest` tags in production. Always specify a fixed version number to ensure consistency and security.

2. **Regular Security Scanning**: Implement regular container security scanning as part of your CI/CD pipeline.

3. **Minimize Base Image Size**: Use smaller base images like Alpine or distroless images to reduce the attack surface.

4. **Multi-stage Builds**: Implement multi-stage builds to reduce final image size and remove build dependencies.

5. **Least Privilege Principle**: Run containers with the minimum required privileges and as non-root users whenever possible.

6. **Update Strategy**: Establish a regular update process for base images and dependencies.

7. **Content Trust**: Consider implementing Docker Content Trust to verify image authenticity.

8. **Resource Limits**: Set resource limits for containers to prevent DoS-style attacks.

## Next Steps

1. Update docker-compose.yml with fixed image versions
2. Update Dockerfile.darf if needed
3. Implement regular security scanning for Docker images
4. Create an update policy for security patches
