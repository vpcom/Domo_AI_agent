from tools.job.run_job_agent import run_job_agent


def create_job_files(job_folder: str):
    yield from run_job_agent(folder_path=job_folder)
