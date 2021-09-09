# pass 'dev' as an arugment to initialize DB for development env.
if [[ $1 == 'dev' ]]; then
    export FLASK_ENV=development
fi

flask db init
flask db migrate
flask db upgrade