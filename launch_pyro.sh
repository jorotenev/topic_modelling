source activate snakes
num_workers=$1
if [ -z "$num_workers" ]; then
	echo "You need to specify workers"
	exit
fi
python -m Pyro4.naming -n 0.0.0.0 &
echo "Launched the server" 

for (( i = 0; i<$num_workers; i++ )); do
	python -m gensim.models.lda_worker &	
done
echo "Launched $num_workers workers."
python -m gensim.models.lda_dispatcher &

touch '.num_workers'
echo "$num_workers" > '.num_workers'
echo "Launched dispatcher."
echo "-----"
echo "You can go play now "