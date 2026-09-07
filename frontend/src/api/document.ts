import { API_BASE_URL, apiJson } from './client'; import type { DocumentScreenshotList, DocumentStatusResponse, DocumentSummary } from '../types/api';
const base=(id:string)=>`/api/jobs/${encodeURIComponent(id)}/document`;
export const getDocumentStatus=(id:string,signal?:AbortSignal)=>apiJson<DocumentStatusResponse>(base(id),{signal});
export const getDocumentSummary=(id:string,signal?:AbortSignal)=>apiJson<DocumentSummary>(`${base(id)}/summary`,{signal});
export const getDocumentScreenshots=(id:string,signal?:AbortSignal,offset=0,limit=200)=>apiJson<DocumentScreenshotList>(`${base(id)}/screenshots?offset=${offset}&limit=${limit}`,{signal});
export const getDocumentDownloadUrl=(id:string)=>`${API_BASE_URL}${base(id)}/download`;
export const getScreenshotPreviewUrl=(id:string,candidateId:number)=>`${API_BASE_URL}${base(id)}/screenshots/${candidateId}/preview`;
