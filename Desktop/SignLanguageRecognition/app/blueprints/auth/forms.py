"""WTForms definitions for the authentication pages."""

from __future__ import annotations

import re

from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, StringField, SubmitField
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    ValidationError,
)

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128


def validate_password_strength(form, field):
    """Require at least eight characters with a letter and a digit."""
    value = field.data or ""
    if len(value) < PASSWORD_MIN_LENGTH:
        raise ValidationError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters long."
        )
    if len(value) > PASSWORD_MAX_LENGTH:
        raise ValidationError(
            f"Password must be at most {PASSWORD_MAX_LENGTH} characters long."
        )
    if not re.search(r"[A-Za-z]", value):
        raise ValidationError("Password must contain at least one letter.")
    if not re.search(r"\d", value):
        raise ValidationError("Password must contain at least one number.")


class LoginForm(FlaskForm):
    email = StringField(
        "Email address", validators=[DataRequired(), Email(), Length(max=180)]
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(max=PASSWORD_MAX_LENGTH)]
    )
    remember = BooleanField("Keep me signed in")
    submit = SubmitField("Sign in")


class RegistrationForm(FlaskForm):
    full_name = StringField(
        "Full name", validators=[DataRequired(), Length(min=2, max=120)]
    )
    email = StringField(
        "Email address", validators=[DataRequired(), Email(), Length(max=180)]
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=PASSWORD_MIN_LENGTH, max=PASSWORD_MAX_LENGTH),
            validate_password_strength,
        ],
    )
    confirm_password = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo("password", message="The passwords do not match."),
        ],
    )
    accept_terms = BooleanField(
        "I understand that only hand-landmark data will be stored and that "
        "validated MSL signs require verification by a qualified signer."
    )
    submit = SubmitField("Create account")


class ProfileForm(FlaskForm):
    full_name = StringField(
        "Full name", validators=[DataRequired(), Length(min=2, max=120)]
    )
    submit = SubmitField("Update profile")
