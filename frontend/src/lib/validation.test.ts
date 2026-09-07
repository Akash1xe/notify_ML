import { describe,expect,it } from 'vitest'; import { isYouTubeUrl } from './validation';
describe('YouTube validation',()=>{it('accepts supported hosts',()=>{expect(isYouTubeUrl(' https://youtu.be/abc ')).toBe(true);expect(isYouTubeUrl('https://www.youtube.com/watch?v=x')).toBe(true)});it('rejects other URLs',()=>expect(isYouTubeUrl('https://example.com/video')).toBe(false))});
