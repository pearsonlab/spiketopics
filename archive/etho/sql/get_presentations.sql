SELECT unitId, trialId, frameNumber, frameClipNumber, movieId 
FROM UnitTrials 
JOIN CleanPresentations2 USING (trialId) 
JOIN FrameEtho USING (trialId) 
JOIN TrialCensor USING (trialId);