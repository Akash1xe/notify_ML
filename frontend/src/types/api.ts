export type JobStatus = 'QUEUED'|'RUNNING'|'COMPLETED'|'FAILED'|'CANCELLED';
export interface JobErrorInfo { code:string; message:string; category?:string|null; failed_stage?:string|null }
export interface Job { id:string; source_url:string; status:JobStatus; stage:string; progress:number; message:string; created_at:string; updated_at:string; error?:JobErrorInfo|null }
export type DocumentStatus = 'NOT_STARTED'|'PROCESSING'|'READY'|'STALE'|'CORRUPT'|'FAILED'|'EMPTY';
export interface DocumentStatusResponse { job_id:string; status:DocumentStatus; checkpoint?:string|null; document_title:string; page_count:number; file_size_bytes:number; final_screenshot_count:number; download_available:boolean; error_code?:string|null }
export interface DocumentSummary { document_title:string; page_count:number; final_screenshot_count:number; suppressed_duplicate_count:number; file_size_bytes:number; first_timestamp_seconds?:number|null; last_timestamp_seconds?:number|null; content_type_counts:Record<string,number> }
export interface DocumentScreenshot { order:number; candidate_id:number; timestamp_seconds:number; timestamp_display?:string|null; content_type:string; width:number; height:number; quality_score:number; semantic_score:number; preview_available:boolean }
export interface DocumentScreenshotList { items:DocumentScreenshot[]; total:number; offset:number; limit:number }
export interface ApiError { code:string; message:string; status:number }
