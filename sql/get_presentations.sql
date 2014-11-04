/*
usage:
msyql -u root surfer < get_presentations.csv > outfile
*/

SELECT unitId, frameNumber, frameClipNumber, movieId 
FROM UnitTrials 
JOIN CleanPresentations2 USING (trialId) 
JOIN FrameEtho USING (trialId) 
JOIN TrialCensor USING (trialId); 
