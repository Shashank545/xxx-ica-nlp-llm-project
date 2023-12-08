# Word Layout Segmenatation

## Introduction

Word document layout segmentation service to get layout information from word file (docx).

---

## Build Environment

Create .env file for passing your PAT for when building the container:

```MP_AZURE_DEVOPS_PERSONAL_ACCESS_TOKEN=<PAT>```

---

### Build & Run container
Run this to build first: `docker compose -f microservices/docker-compose.yaml --env-file .env build --progress=plain --pull`

Then you should be able to run it with: `docker compose -f microservices/docker-compose.yaml --env-file .env up `

Access Swagger UI at for local testing http://localhost:8000/docs

The framework will detect that you are running on local and you can try out changes easily.

## Getting async job result.
### Nowait in getting result form async job.

`Nowait` is false by default.
Setting it to true will keep the connection alive for:
`LONG_POLLING_TIMEOUT_IN_SECONDS` and will retry every
`POLLING_INTERVAL_IN_SECONDS` until returning result and closing the connection.

At the moment these values are hard-coded to:
`POLLING_INTERVAL_IN_SECONDS`: 5
`LONG_POLLING_TIMEOUT_IN_SECONDS`: 30

---

### Tip about x-verbosity-level
To test these levels above, simple raise an exception
that's not caught in `AsyncJobWorker` method.

### x-verbosity-level: 0
- Just worker failed in result.

### x-verbosity-level: 1
- Worker failed.
- Traceback of exception.

### x-verbosity-level: 2
- Worker failed.
- Traceback of exception.
- Job history.
