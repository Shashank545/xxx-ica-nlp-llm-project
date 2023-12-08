import io
from logging import getLogger

from document_layout_segmentation.pdf_document_segmentation import (
    layout_analysis,
)
from document_layout_segmentation.word_document_segmentation import (
    word_segmentation,
)
from fastapi import UploadFile
from ica_inference.utils.common_utils import pdf_char_extractor
from kitmicroservices.framework.jobcontext import (
    AsyncJobContext,
    SyncJobContext,
)
from kitmicroservices.framework.microservice import Microservice
from starlette.responses import JSONResponse

Service = Microservice(
    title="Document Layout segmentation",
    description="Document segmentation functionality",
)

logger = getLogger(__name__)


class WrongFileException(Exception):
    pass


class WordSegmentationException(Exception):
    pass


class PDFSegmentationException(Exception):
    pass


class MRSegmentationException(Exception):
    pass


@Service.SyncJobEndpoint(method="POST", path="/word_to_json")
def word_to_json_sync(sync_job_context: SyncJobContext, word_file: UploadFile):
    try:
        output = run_word_segmentation(word_file)
        sync_job_context.append_workload_info(
            {"unit_id": "word_file_segmentation", "unit_total": 1}
        )

        return JSONResponse(
            content={
                "data": output,
                "message": "Word segmentation is done!",
                "status": "Complete",
            },
            status_code=200,
        )
    except WrongFileException:
        return JSONResponse(
            {
                "data": {},
                "message": f"Expected word file, got: {word_file.content_type}",
                "status": "Failed",
            },
            status_code=400,
        )
    except Exception:
        logger.exception("Word segmentation failed.")
        return JSONResponse(
            content={
                "data": {},
                "message": "Word segmentation failed!",
                "status": "Failed",
            },
            status_code=400,
        )


@Service.AsyncJobSubmission(name="word_to_json_async")
async def word_to_json_async(
    async_job_context: AsyncJobContext, word_file: UploadFile
):
    data = {"word_file": word_file}
    await async_job_context.save_parameters(data)
    async_job_context.set_timeout_in_minutes(
        AsyncJobContext.TimeoutInMinutes.SHORT
    )


@Service.AsyncJobResult(name="word_to_json_async")
async def word_to_json_async_result(async_job_context: AsyncJobContext) -> dict:
    data: dict = await async_job_context.get_results()
    return data


@Service.AsyncJobWorker(name="word_to_json_async")
async def word_to_json_worker(async_job_context: AsyncJobContext) -> None:
    try:
        parameters = await async_job_context.get_parameters()
        word_file = parameters.get("word_file", None)
        output = run_word_segmentation(word_file)
    except Exception:
        raise WordSegmentationException("Worker failed to do word segmentation")
    else:
        async_job_context.append_workload_info(
            {"unit_id": "word_file_segmentation", "unit_total": 1}
        )
        await async_job_context.save_results(output)


@Service.AsyncJobSubmission(name="pdf_to_json_async")
async def pdf_to_json_async(
    async_job_context: AsyncJobContext, pdf_file: UploadFile, document_lang: str
):
    data = {"pdf_file": pdf_file, "document_lang": document_lang}
    await async_job_context.save_parameters(data)
    async_job_context.set_timeout_in_minutes(
        AsyncJobContext.TimeoutInMinutes.SHORT
    )


@Service.AsyncJobResult(name="pdf_to_json_async")
async def pdf_to_json_async_result(async_job_context: AsyncJobContext) -> dict:
    data: dict = await async_job_context.get_results()
    return data


@Service.AsyncJobWorker(name="pdf_to_json_async")
async def pdf_to_json_worker(async_job_context: AsyncJobContext) -> None:
    try:
        parameters = await async_job_context.get_parameters()
        pdf_file = parameters.get("pdf_file", None)

        document_lang = parameters.get("document_lang", None)
        output = run_pdf_segmentation(
            pdf_file=pdf_file, document_lang=document_lang
        )
    except Exception:
        raise PDFSegmentationException("Worker failed to do pdf segmentation")
    else:
        async_job_context.append_workload_info(
            {"unit_id": "pdf_file_segmentation", "unit_total": 1}
        )
        await async_job_context.save_results(output)


def is_word_file(word_file: UploadFile):
    if word_file.content_type in (
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # noqa: E501
    ):
        return True
    else:
        return False


def is_pdf_file(pdf_file: UploadFile):
    if pdf_file.content_type in [
        'application/pdf',
    ]:
        return True
    else:
        return False


def run_word_segmentation(word_file: UploadFile):
    temp_buffer = io.BytesIO()
    try:
        temp_buffer.write(word_file.file.read())
        temp_buffer.seek(0)
        return word_segmentation(temp_buffer)
    finally:
        word_file.close()
        temp_buffer.close()


def run_pdf_segmentation(pdf_file: UploadFile, document_lang: str):
    temp_buffer = io.BytesIO()
    try:
        temp_buffer.write(pdf_file.file.read())
        temp_buffer.seek(0)
        return layout_analysis(temp_buffer, document_lang)
    finally:
        pdf_file.close()
        temp_buffer.close()


def run_mr_pdf_document_layout(pdf_file: UploadFile):
    """
    This is for running layout analysis
    """
    temp_buffer = io.BytesIO()
    try:
        temp_buffer.write(pdf_file.file.read())
        temp_buffer.seek(0)
        return pdf_char_extractor(temp_buffer)
    finally:
        pdf_file.close()
        temp_buffer.close()


@Service.AsyncJobSubmission(name="mr_pdf_document_layout")
async def mr_pdf_document_layout_async(
    async_job_context: AsyncJobContext, pdf_file: UploadFile
):
    data = {"pdf_file": pdf_file}
    await async_job_context.save_parameters(data)
    async_job_context.set_timeout_in_minutes(
        AsyncJobContext.TimeoutInMinutes.SHORT
    )


@Service.AsyncJobResult(name="mr_pdf_document_layout")
async def mr_pdf_document_layout_result(
    async_job_context: AsyncJobContext,
) -> dict:
    data: dict = await async_job_context.get_results()
    return data


@Service.AsyncJobWorker(name="mr_pdf_document_layout")
async def mr_pdf_document_layout_worker(
    async_job_context: AsyncJobContext,
) -> None:
    try:
        parameters = await async_job_context.get_parameters()
        pdf_file = parameters.get("pdf_file", None)

        output = run_mr_pdf_document_layout(pdf_file=pdf_file)
        output = {"layout-result": output}
    except Exception:
        raise MRSegmentationException(
            "Worker failed to do machine readable pdf segmentation"
        )
    else:
        async_job_context.append_workload_info(
            {"unit_id": "mr_pdf_document_layout_segmentation", "unit_total": 1}
        )
        await async_job_context.save_results(output)


@Service.SyncJobEndpoint(method="POST", path="/mr_pdf_document_layout_sync")
def mr_pdf_document_layout_sync(
    sync_job_context: SyncJobContext, pdf_file: UploadFile
):
    try:
        output = run_mr_pdf_document_layout(pdf_file)
        sync_job_context.append_workload_info(
            {"unit_id": "mr_pdf_document_layout_segmentation", "unit_total": 1}
        )

        return JSONResponse(
            content={
                "data": output,
                "message": "PDF document segmentation is done!",
                "status": "Complete",
            },
            status_code=200,
        )
    except WrongFileException:
        return JSONResponse(
            {
                "data": {},
                "message": f"Expected PDF file, got: {pdf_file.content_type}",
                "status": "Failed",
            },
            status_code=400,
        )
    except Exception:
        logger.exception("PDF document segmentation failed.")
        return JSONResponse(
            content={
                "data": {},
                "message": "PDF document segmentation failed!",
                "status": "Failed",
            },
            status_code=400,
        )
