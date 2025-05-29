import string
from typing import Dict, Any, List, Union, Set, Optional
from vllm_judge.models import TemplateEngine
from vllm_judge.exceptions import InvalidInputError


class TemplateProcessor:
    """Template processing for dynamic prompts. 
    Handles template variable substitution."""
    
    @staticmethod
    def apply_template(
        template: Optional[Union[str, Dict]],
        template_vars: Dict[str, Any],
        engine: TemplateEngine = TemplateEngine.FORMAT,
        strict: bool = True
    ) -> Optional[Union[str, Dict]]:
        """
        Apply template variables to a template string or dict.
        
        Args:
            template: Template string, dict, or None
            template_vars: Variables to substitute
            engine: Template engine to use
            strict: If True, raise error for missing variables
            
        Returns:
            Processed template
            
        Raises:
            InvalidInputError: If required variables are missing
        """
        if isinstance(template, dict):
            # Process dict values recursively
            return {
                k: TemplateProcessor.apply_template(v, template_vars, engine, strict)
                for k, v in template.items()
            }
        
        if not isinstance(template, str):
            return template
        
        if engine == TemplateEngine.FORMAT:
            return TemplateProcessor._apply_format_template(
                template, template_vars, strict
            )
        elif engine == TemplateEngine.JINJA2:
            return TemplateProcessor._apply_jinja2_template(
                template, template_vars, strict
            )
    
    @staticmethod
    def _apply_format_template(
        template: str,
        template_vars: Dict[str, Any],
        strict: bool
    ) -> str:
        """Apply str.format() style template."""
        try:
            # First check for missing variables if strict
            if strict:
                missing = TemplateProcessor.get_required_vars_format(template) - set(template_vars.keys())
                if missing:
                    raise InvalidInputError(
                        f"Missing required template variables: {', '.join(sorted(missing))}"
                    )
            
            return template.format(**template_vars)
        except KeyError as e:
            if strict:
                raise InvalidInputError(f"Missing template variable: {e}")
            else:
                # Partial formatting - leave missing variables as-is
                return template.format_map(SafeDict(template_vars))
    
    @staticmethod
    def _apply_jinja2_template(
        template: str,
        template_vars: Dict[str, Any],
        strict: bool
    ) -> str:
        """Apply Jinja2 template."""
        try:
            from jinja2 import Template, Environment, StrictUndefined, UndefinedError
        except ImportError:
            raise ImportError(
                "Jinja2 is required for jinja2 template engine. "
                "Install with: pip install vllm-judge[jinja2]"
            )
        
        try:
            if strict:
                # Use StrictUndefined to catch missing variables
                env = Environment(undefined=StrictUndefined)
                jinja_template = env.from_string(template)
            else:
                # Default behavior - missing variables render as empty
                jinja_template = Template(template)
            
            return jinja_template.render(**template_vars)
        except UndefinedError as e:
            raise InvalidInputError(f"Missing template variable in Jinja2 template: {e}")
    
    @staticmethod
    def get_required_vars(
        template: Union[str, Dict, None],
        engine: TemplateEngine = TemplateEngine.FORMAT
    ) -> Set[str]:
        """
        Extract required variables from a template.
        
        Args:
            template: Template to analyze
            engine: Template engine being used
            
        Returns:
            Set of required variable names
        """
        if isinstance(template, dict):
            # Collect from all dict values
            all_vars = set()
            for v in template.values():
                all_vars.update(TemplateProcessor.get_required_vars(v, engine))
            return all_vars
        
        if not isinstance(template, str):
            return set()
        
        if engine == TemplateEngine.FORMAT:
            return TemplateProcessor.get_required_vars_format(template)
        elif engine == TemplateEngine.JINJA2:
            return TemplateProcessor.get_required_vars_jinja2(template)
    
    @staticmethod
    def get_required_vars_format(template: str) -> Set[str]:
        """Extract variables from format string."""
        formatter = string.Formatter()
        variables = set()
        
        try:
            for _, field_name, _, _ in formatter.parse(template):
                if field_name:
                    # Handle nested fields like {user.name}
                    base_var = field_name.split('.')[0].split('[')[0]
                    variables.add(base_var)
        except:
            pass  # If parsing fails, return empty set
        
        return variables
    
    @staticmethod
    def get_required_vars_jinja2(template: str) -> Set[str]:
        """Extract variables from Jinja2 template."""
        try:
            from jinja2 import Environment, meta
        except ImportError:
            return set()  # Can't analyze without Jinja2
        
        try:
            env = Environment()
            ast = env.parse(template)
            return meta.find_undeclared_variables(ast)
        except:
            return set()
    
    @staticmethod
    def validate_template_vars(
        provided_vars: Dict[str, Any],
        required_vars: List[str],
        template_defaults: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate and merge template variables.
        
        Args:
            provided_vars: User-provided variables
            required_vars: Required variable names
            template_defaults: Default values
            
        Returns:
            Merged template variables
            
        Raises:
            InvalidInputError: If required variables are missing
        """
        # Start with defaults
        final_vars = dict(template_defaults or {})
        
        # Override with provided vars
        final_vars.update(provided_vars)
        
        # Check required vars
        missing = set(required_vars) - set(final_vars.keys())
        if missing:
            raise InvalidInputError(
                f"Missing required template variables: {', '.join(sorted(missing))}"
            )
        
        return final_vars


class SafeDict(dict):
    """Dictionary that returns {key} for missing keys in format strings."""
    
    def __missing__(self, key):
        return f"{{{key}}}"