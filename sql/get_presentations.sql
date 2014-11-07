/*
usage:
msyql -u root surfer < get_presentations.sql > outfile
then:
tr "\t" "," < outfile > csvfile
to translate tsv to csv
*/

SELECT unitId, frameNumber, frameClipNumber, movieId 
FROM UnitTrials 
JOIN CleanPresentations2 USING (trialId) 
JOIN FrameEtho USING (trialId) 
JOIN TrialCensor USING (trialId);