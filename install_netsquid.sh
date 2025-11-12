read -p "Enter username: " username
read -s -p "Enter password: " pwd
echo

python -m pip install --extra-index-url https://$username:$pwd@pypi.netsquid.org netsquid