# Execute following line on bash prompt manually
# does not work as a shell script
# http://stackoverflow.com/questions/16630232/sed-copy-substring-from-fixed-position-and-copy-it-in-front-of-line

cat IE_true.seq | awk '{print $2}' | while read line; do echo ${line:58:13}; echo ${line:58:13}>>authss_3prime.data; done;
